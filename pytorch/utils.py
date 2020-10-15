# implementation of tensorboard logger from
# https://becominghuman.ai/logging-in-tensorboard-with-pytorch-or-any-other-library-c549163dee9e

import io
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import atexit

class Tensorboard:
    def __init__(self, logdir):
        self.writer = SummaryWriter(logdir)
        atexit.register(self.close)

    def close(self):
        if self.writer:
            self.writer.close()
        self.writer = None

    def log_scalar(self, tag, value, global_step):
        assert self.writer, "Writer is closed!"
        self.writer.add_scalar(tag, value, global_step=global_step)
        self.writer.flush()

    def log_histogram(self, tag, values, global_step, bins):
        assert self.writer, "Writer is closed!"
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_image(self, tag, img, global_step):
        assert self.writer, "Writer is closed!"
        s = io.BytesIO()
        Image.fromarray(img).save(s, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_plot(self, tag, figure, global_step):
        assert self.writer, "Writer is closed!"
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()
