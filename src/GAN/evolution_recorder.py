import tempfile
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image
from fastcore.basics import store_attr
from fastai.callback.core import Callback

class GANEvolutionRecorder(Callback):
    "`Callback` that records the evolution of the generator during the training process."
    def __init__(self, test_batch=None, every_k=1, max_n=9, storege_path=None, show_after_fit=False, figsize=None, plot_fn=None):
        store_attr('test_batch,every_k,max_n,storege_path,show_after_fit,figsize,plot_fn')
        # FIXME
        #if self.test_batch is not None:
        #    self.test_batch = self.learn.gen_batch(self.test_batch)
        #    #self.test_batch = test_batch[:-1] # HACK this may not work in data another setup. Needs to be generalised?
        self.temp_dir = False if self.storege_path else True
        if self.storege_path:
          os.makedirs(self.storege_path, exist_ok=True)
        else:
          self.storege_path = tempfile.mkdtemp()
        self.records = 0

    def __del__(self):
        if self.temp_dir:
            shutil.rmtree(self.storege_path)

    def file_path(self, record):
        "Get path to save the generator output."
        fn = 'record_{}.png'.format(record)
        return os.path.join(self.storege_path , fn)

    def before_fit(self):
        "Set the batch for tracking the generator evolution if it is not provided."
        if self.test_batch is None:
            if self.dls.valid.n > 0:
                batch = self.dls.valid.one_batch()
            else:
                batch = self.dls.train.one_batch()
            self.test_batch = self.learn.gen_batch(batch)
            #self.test_batch = batch[:-1] # HACK
            t = list()
            for i in range(len(self.test_batch)):
                t.append(self.test_batch[i][:self.max_n])
            self.test_batch = tuple(t)

    def after_epoch(self):
        "Save the output of current generator."
        # FIXME avoid running this code after Learner.show_results and Learner.get_preds
        if self.records%self.every_k == self.every_k-1:
            output = self.generate(self.test_batch)[0]
            path = self.file_path(self.records)
            if self.plot_fn:
                self.plot_fn(path, self.test_batch, output)
            else:
                save_image(output, path)
        self.records += 1

    def after_fit(self):
        "Plot the evolution of the generator if needed."
        if self.show_after_fit:
            self.plot_animation()

    def create_animation(self, figsize=None):
        "Create matplotlib.animation representing the evolution of the generator."
        # WARNING this will probably crash your notebook if you have too many images to show or they are too big
        if figsize is None:
            figsize = self.figsize
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(left=0.0,
                            bottom=0.0,
                            right=1.0,
                            top=1.0)
        plt.axis("off")
        ims = []
        for i in range(self.records):
            if i%self.every_k == self.every_k-1:
                arr = plt.imread(self.file_path(i))
                im = plt.imshow(arr, animated=True)
                ims.append([im])
        plt.close()
        return animation.ArtistAnimation(fig, ims)

    def plot_animation(self, figsize=None):
        "Embed the evolution animation in Jupyter notebook."
        ani = self.create_animation(figsize)
        try:
            from IPython.display import display, HTML
        except:
            warn(f"Cannot import display, HTML from IPython.display.")
            return
        display(HTML(ani.to_jshtml())) 
