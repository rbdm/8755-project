# Adapted from https://github.com/wandb/client/tree/master/wandb/integration/fastai
# Adapted to add latent_z at inference time, for MSGAN and GAN runs
# For MSGAN runs, use capture_mode='MS'
# For GAN runs, use capture_mode='GAN'
# For regular runs, use capture_mode='other'

import wandb
import fastai
from fastai.callbacks import TrackerCallback
from pathlib import Path
import random
import torch
from fastai.vision import Image as fastaiIm

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend (avoid tkinter issues)
    import matplotlib.pyplot as plt
except:
    print("Warning: matplotlib required if logging sample image predictions")


class WandbCallback(TrackerCallback):
    """
    Automatically saves model topology, losses & metrics.
    Optionally logs weights, gradients, sample predictions and best trained model.
    Arguments:
        learn (fastai.basic_train.Learner): the fast.ai learner to hook.
        log (str): "gradients", "parameters", "all", or None. Losses & metrics are always logged.
        save_model (bool): savhttps://github.com/wandb/client/tree/master/wandb/integration/fastaie model at the end of each epoch. It will also load best model at the end of training.
        monitor (str): metric to monitor for saving best model. None uses default TrackerCallback monitor value.
        mode (str): "auto", "min" or "max" to compare "monitor" values and define best model.
        input_type (str): "images" or None. Used to display sample predictions.
        validation_data (list): data used for sample predictions if input_type is set.
        predictions (int): number of predictions to make if input_type is set and validation_data is None.
        seed (int): initialize random generator for sample predictions if input_type is set and validation_data is None.
    """

    # Record if watch has been called previously (even in another instance)
    _watch_called = False

    def __init__(
        self,
        learn,
        log="gradients",
        save_model=True,
        monitor=None,
        mode="auto",
        input_type=None,
        validation_data=None,
        predictions=36,
        seed=12345,
        capture_mode="other",
    ):

        # Check if wandb.init has been called
        if wandb.run is None:
            raise ValueError("You must call wandb.init() before WandbCallback()")

        # Adapted from fast.ai "SaveModelCallback"
        if monitor is None:
            # use default TrackerCallback monitor value
            super().__init__(learn, mode=mode)
        else:
            super().__init__(learn, monitor=monitor, mode=mode)
        self.save_model = save_model
        self.model_path = Path(wandb.run.dir) / "bestmodel.pth"

        self.log = log
        self.input_type = input_type
        self.best = None
        self.capture_mode = capture_mode

        # Select items for sample predictions to see evolution along training
        self.validation_data = validation_data
        if input_type and not self.validation_data:
            wandbRandom = random.Random(seed)  # For repeatability
            predictions = min(predictions, len(learn.data.valid_ds))
            indices = wandbRandom.sample(range(len(learn.data.valid_ds)), predictions)
            self.validation_data = [learn.data.valid_ds[i] for i in indices]

    def on_train_begin(self, **kwargs):
        "Call watch method to log model topology, gradients & weights"

        # Set self.best, method inherited from "TrackerCallback" by "SaveModelCallback"
        super().on_train_begin()

        # Ensure we don't call "watch" multiple times
        if not WandbCallback._watch_called:
            WandbCallback._watch_called = True

            # Logs model topology and optionally gradients and weights
            wandb.watch(self.learn.model, log=self.log)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        "Logs training loss, validation loss and custom metrics & log prediction samples & save model"

        if self.save_model:
            # Adapted from fast.ai "SaveModelCallback"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(
                    "Better model found at epoch {} with {} value: {}.".format(
                        epoch, self.monitor, current
                    )
                )
                self.best = current

                # Save within wandb folder
                with self.model_path.open("wb") as model_file:
                    self.learn.save(model_file)

        # Log sample predictions if learn.predict is available
        if self.validation_data:
            try:
                self._wandb_log_predictions()
                print("current epoch :"+str(epoch))
            except FastaiError as e:
                wandb.termwarn(e.message)
                #self.validation_data = None  # prevent from trying again on next loop
            except Exception as e:
                wandb.termwarn("Unable to log prediction samples.\n{}".format(e))
                #self.validation_data = None  # prevent from trying again on next loop

        # Log losses & metrics
        # Adapted from fast.ai "CSVLogger"
        logs = {
            name: stat
            for name, stat in list(
                zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)
            )
        }
        wandb.log(logs)

    def on_train_end(self, **kwargs):
        "Load the best model."

        if self.save_model:
            # Adapted from fast.ai "SaveModelCallback"
            if self.model_path.is_file():
                with self.model_path.open("rb") as model_file:
                    self.learn.load(model_file, purge=False)
                    print("Loaded best saved model from {}".format(self.model_path))
    
    # get random latent z for logging MSGAN training
    def get_z_random_single(self, nz, normalize=True):
        # create random latent z as conditioning input
        z = torch.FloatTensor(nz)
        z.copy_(torch.randn(nz))
        if normalize:
            z -= z.min()
            z /= z.max()
        return z
                    
    def _wandb_log_predictions(self):
        "Log prediction samples"

        pred_log = []

        for x, y in self.validation_data:
            
            # for MSGAN runs, add latent z to x at inference
            if (self.capture_mode=="MS"):
                z_rand = self.get_z_random_single(8)
                z_exp = z_rand.view(len(z_rand), 1, 1)
                z_exp = z_exp.expand(len(z_rand), x.data.size(1), x.data.size(2))
                x_cond = torch.cat((x.data, z_exp), 0).unsqueeze(0)
                pred = self.learn.model.generator(x_cond.cuda()).detach().cpu().squeeze()
                pred -= pred.min()#pred.min()
                pred /= pred.max()
                pred_im = fastaiIm(pred)
                
                
            elif self.capture_mode=="GAN":
                pred = self.generator.learn.predict(x)
                
            else:
                try:
                    pred = self.learn.predict(x)
                    print("saved prediction samples through wandb")
                except:
                    pass
                    
            # scalar -> likely to be a category
            # tensor of dim 1 -> likely to be multicategory
            if not pred[1].shape or pred[1].dim() == 1:
                pred_log.append(
                    wandb.Image(
                        x.data,
                        caption="Ground Truth: {}\nPrediction: {}".format(y, pred[0]),
                    )
                )

            # most vision datasets have a "show" function we can use
            elif hasattr(x, "show"):
                # log input data
                pred_log.append(wandb.Image(x.data, caption="Input data", grouping=3))

                # log label and prediction
                #print("here")
                for im, capt in ((pred[0], "Prediction"), (y, "Ground Truth")):
                    if not hasattr(pred[0], "show"):
                        pred[0] = fastaiIm(pred[0])
                    # Resize plot to image resolution
                    # from https://stackoverflow.com/a/13714915
                    my_dpi = 100
                    fig = plt.figure(frameon=False, dpi=my_dpi)
                    h, w = x.size
                    fig.set_size_inches(w / my_dpi, h / my_dpi)
                    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                    ax.set_axis_off()
                    fig.add_axes(ax)

                    # Superpose label or prediction to input image
                    x.show(ax=ax, y=im)
                    pred_log.append(wandb.Image(fig, caption=capt))
                    plt.close(fig)

            # likely to be an image
            elif hasattr(y, "shape") and (
                (len(y.shape) == 2) or (len(y.shape) == 3 and y.shape[0] in [1, 3, 4])
            ):
                pred_log.extend(
                    [
                        wandb.Image(x.data, caption="Input data", grouping=3),
                        wandb.Image(pred[0].data, caption="Prediction"),
                        wandb.Image(y.data, caption="Ground Truth"),
                    ]
                )

            # else we just log input data
            else:
                pred_log.append(wandb.Image(x.data, caption="Input data"))

            wandb.log({"Prediction Samples": pred_log}, commit=False)


class FastaiError(wandb.Error):
    pass