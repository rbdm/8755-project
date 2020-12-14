from fastai.torch_core import *
from fastai.layers import *
from fastai.callback import *
from fastai.basic_data import *
from fastai.basic_train import Learner, LearnerCallback
from fastai.vision.image import Image
from fastai.vision.data import ImageList
from deoldify.loss import FeatureLossMS
from fastai.vision.gan import gan_loss_from_func, GANModule, FixedGANSwitcher
import random

class GANLoss_MS(GANModule):
    "Wrapper around `loss_funcC` (for the critic) and `loss_funcG` (for the generator)."
    def __init__(self, loss_funcG:Callable, loss_funcC:Callable, gan_model:GANModule):
        super().__init__()
        self.loss_funcG,self.loss_funcC,self.gan_model = loss_funcG,loss_funcC,gan_model

    def generator(self, output, target):
        "Evaluate the `output` with the critic then uses `self.loss_funcG` to combine it with `target`."
        
        target = torch.cat((target,target),0)
        fake_pred = self.gan_model.critic(output)
        
        return self.loss_funcG(fake_pred, target, output)

    def critic(self, real_pred, input):
        "Create some `fake_pred` with the generator from `input` and compare them to `real_pred` in `self.loss_funcD`."

        z_random = get_z_random(input.size(0), 8, 'gaussian')
        z_random = z_random.view(z_random.size(0), z_random.size(1),1,1).expand(z_random.size(0), z_random.size(1), input.size(2), input.size(3))
        
        if (input.dtype==torch.float16): 
            z_random=to_half(z_random)
        
        conditioned_input = torch.cat([input, z_random], 1)
        
        
        fake = self.gan_model.generator(conditioned_input.requires_grad_(False)).requires_grad_(True)
        
        fake_pred = self.gan_model.critic(fake)
        
        return self.loss_funcC(real_pred, fake_pred)

class GANTrainer_MS(LearnerCallback):
    "Handles GAN Training."
    _order=-20
    def __init__(self, learn:Learner, switch_eval:bool=False, clip:float=None, beta:float=0.98, gen_first:bool=False,
                 show_img:bool=True, nz:int=8):
        super().__init__(learn)
        self.switch_eval,self.clip,self.beta,self.gen_first,self.show_img = switch_eval,clip,beta,gen_first,show_img
        self.generator,self.critic = self.model.generator,self.model.critic
        self.nz = nz
        self.crit_mode=False

    def _set_trainable(self):
        train_model = self.generator if     self.gen_mode else self.critic
        loss_model  = self.generator if not self.gen_mode else self.critic
        requires_grad(train_model, True)
        requires_grad(loss_model, False)
        if self.switch_eval:
            train_model.train()
            loss_model.eval()

    def on_train_begin(self, **kwargs):
        "Create the optimizers for the generator and critic if necessary, initialize smootheners."
        if not getattr(self,'opt_gen',None):
            self.opt_gen = self.opt.new([nn.Sequential(*flatten_model(self.generator))])
        else: self.opt_gen.lr,self.opt_gen.wd = self.opt.lr,self.opt.wd
        if not getattr(self,'opt_critic',None):
            self.opt_critic = self.opt.new([nn.Sequential(*flatten_model(self.critic))])
        else: self.opt_critic.lr,self.opt_critic.wd = self.opt.lr,self.opt.wd
        self.gen_mode = self.gen_first
        self.switch(self.gen_mode)
        self.closses,self.glosses = [],[]
        self.smoothenerG,self.smoothenerC = SmoothenValue(self.beta),SmoothenValue(self.beta)
        self.recorder.add_metric_names(['crit_loss', 'gen_loss', 'gen_MS_loss'])
        self.imgs,self.titles = [],[]
        

    def on_train_end(self, **kwargs):
        "Switch in generator mode for showing results."
        self.switch(gen_mode=True)

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "Clamp the weights with `self.clip` if it's not None, return the correct input."
        if self.clip is not None:
            for p in self.critic.parameters(): p.data.clamp_(-self.clip, self.clip)
        if last_input.dtype==torch.float16: 
            last_target=to_half(last_target)
            
        if self.gen_mode:
            self.z_random_1 = get_z_random(last_input.size(0), self.nz, 'gaussian')
            self.z_random_2 = get_z_random(last_input.size(0), self.nz, 'gaussian')
            
            z_conc = torch.cat((self.z_random_1, self.z_random_2),0)#.cuda()
            in_conc = torch.cat((last_input, last_input),0)#.cuda()
            z_expanded = z_conc.view(z_conc.size(0), z_conc.size(1), 1, 1)
            z_expanded = z_expanded.expand(z_conc.size(0), z_conc.size(1), in_conc.size(2), in_conc.size(3))
            
            if last_input.dtype==torch.float16:
                z_expanded=to_half(z_expanded)
            
            last_input = torch.cat([in_conc, z_expanded], 1)
            
            return {'last_input':last_input,'last_target':last_target}
    
        else:
            return {'last_input':last_target,'last_target':last_input}
        
    def on_backward_begin(self, last_loss, last_output, **kwargs):
        "Record `last_loss` in the proper list."
        
        if self.gen_mode:
            if (len(last_output.shape)==4):
                self.out_random_1, self.out_random_2 = torch.split(last_output, self.z_random_1.size(0), dim=0)
                            
            loss_z = torch.mean(torch.abs(self.out_random_1 - self.out_random_2)) / torch.mean(torch.abs(self.z_random_1 - self.z_random_2))
            self.ms_regularization = 1 / (loss_z + 1e-6) #prevents division by 0
            
            self.smoothenerG.add_value(last_loss.detach().cpu() + self.ms_regularization.detach().cpu())
            self.glosses.append(self.smoothenerG.smooth)
            self.last_gen = last_output.detach().cpu()
            return {'last_loss': (last_loss.cuda() + self.ms_regularization.cuda())}
        else:
            self.smoothenerC.add_value(last_loss)
            self.closses.append(self.smoothenerC.smooth)
    
    def on_epoch_begin(self, epoch, **kwargs):
        "Put the critic or the generator back to eval if necessary."
        self.switch(self.gen_mode)

    def on_epoch_end(self, pbar, epoch, last_metrics, **kwargs):
        self.z_random_1 = None
        self.z_random_2 = None
        self.out_random_1 = None
        self.out_random_2 = None
        gc.collect()
        
        "Put the various losses in the recorder and show a sample image."
        if not hasattr(self, 'last_gen') or not self.show_img: return
        data = self.learn.data
        
        if (random.uniform(0,1)>=0.5):
            img = self.last_gen[0]
        else:
            img = self.last_gen[1]
        norm = getattr(data,'norm',False)
        if norm and norm.keywords.get('do_y',False): img = data.denorm(img)
        img = data.train_ds.y.reconstruct(img)
        self.imgs.append(img)
        self.titles.append(f'Epoch {epoch}')
        pbar.show_imgs(self.imgs, self.titles)
        
        return add_metrics(last_metrics, [getattr(self.smoothenerC,'smooth',None), getattr(self.smoothenerG,'smooth',None), self.ms_regularization.detach().cpu()])

    def switch(self, gen_mode:bool=None):
        "Switch the model, if `gen_mode` is provided, in the desired mode."
        self.gen_mode = (not self.gen_mode) if gen_mode is None else gen_mode
        self.opt.opt = self.opt_gen.opt if self.gen_mode else self.opt_critic.opt
        self._set_trainable()
        self.model.switch(gen_mode)
        self.loss_func.switch(gen_mode)


def get_z_random(bs, nz, rand_type='gaussian', normalize=True):
    z = torch.FloatTensor(bs, nz)
    if rand_type=='gaussian':
        z.copy_(torch.randn(bs, nz))
    else:
        z = torch.rand(batch_size, nz)   
    if normalize:
        z -= z.min()
        z /= z.max()
    return z.cuda()

class GANLearner_MS(Learner):
    "Customised for MSGAN"
    def __init__(self, data:DataBunch, generator:nn.Module, critic:nn.Module, gen_loss_func:LossFunction,
                 crit_loss_func:LossFunction, switcher:Callback=None, gen_first:bool=False, switch_eval:bool=True,
                 show_img:bool=True, clip:float=None, nz:int=8, **learn_kwargs):
        gan = GANModule(generator, critic)
        self.nz = nz
        loss_func = GANLoss_MS(gen_loss_func, crit_loss_func, gan)
        switcher = ifnone(switcher, partial(FixedGANSwitcher, n_crit=5, n_gen=1))
        super().__init__(data, gan, loss_func=loss_func, callback_fns=[switcher], **learn_kwargs)
        trainer = GANTrainer_MS(self, clip=clip, switch_eval=switch_eval, show_img=show_img, nz=nz, gen_first=True)
        self.gan_trainer = trainer
        self.callbacks.append(trainer)

    @classmethod
    def from_learners(cls, learn_gen:Learner, learn_crit:Learner, switcher:Callback=None,
                      weights_gen:Tuple[float,float]=None, **learn_kwargs):
        "Create a GAN from `learn_gen` and `learn_crit`."
        losses = gan_loss_from_func(learn_gen.loss_func, learn_crit.loss_func, weights_gen=weights_gen)
        return cls(learn_gen.data, learn_gen.model, learn_crit.model, *losses, switcher=switcher, **learn_kwargs)