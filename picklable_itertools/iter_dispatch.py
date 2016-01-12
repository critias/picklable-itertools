import io
import gzip
import six
try:
    import numpy
    NUMPY_AVAILABLE = True
except ImportError:
    numpy = None
    NUMPY_AVAILABLE = False


from .base import BaseItertool


def iter_(obj):
    """A custom replacement for iter(), dispatching a few custom picklable
    iterators for known types.
    """
    if six.PY2:
        file_types = file,  # noqa
    if six.PY3:
        file_types = io.IOBase,
        dict_items = {}.items().__class__
        dict_values = {}.values().__class__
        dict_keys = {}.keys().__class__
        dict_view = (dict_items, dict_values, dict_keys)

    if isinstance(obj, dict):
        return ordered_sequence_iterator(list(obj.keys()))
    if isinstance(obj, gzip.GzipFile):
        return gfile_iterator(obj)
    if isinstance(obj, file_types):
        return file_iterator(obj)
    if six.PY2:
        if isinstance(obj, (list, tuple)):
            return ordered_sequence_iterator(obj)
        if isinstance(obj, xrange):  # noqa
            return range_iterator(obj)
        if NUMPY_AVAILABLE and isinstance(obj, numpy.ndarray):
            return ordered_sequence_iterator(obj)
    if six.PY3 and isinstance(obj, dict_view):
        return ordered_sequence_iterator(list(obj))
    return iter(obj)


class range_iterator(BaseItertool):
    """A picklable range iterator for Python 2."""
    def __init__(self, xrange_):
        self._start, self._stop, self._step = xrange_.__reduce__()[1]
        self._n = self._start

    def __next__(self):
        if (self._step > 0 and self._n < self._stop or
                self._step < 0 and self._n > self._stop):
            value = self._n
            self._n += self._step
            return value
        else:
            raise StopIteration


class file_iterator(BaseItertool):
    """A picklable file iterator."""
    def __init__(self, f):
        self._f = f

    def __next__(self):
        line = self._f.readline()
        if not line:
            raise StopIteration
        return line

    def __getstate__(self):
        name, pos, mode = self._f.name, self._f.tell(), self._f.mode
        return name, pos, mode

    def __setstate__(self, state):
        name, pos, mode = state
        self._f = open(name, mode=mode)
        self._f.seek(pos)


class gfile_iterator(file_iterator):
    """A picklable gzip file iterator."""
    def __getstate__(self):
        name, pos, mode = self._f.name, self._f.tell(), self._f.myfileobj.mode
        return name, pos, mode

    def __setstate__(self, state):
        name, pos, mode = state
        self._f = gzip.open(name, mode=mode)
        self._f.seek(pos)


class codecs_iterator(BaseItertool):
    """A picklable codecs stream iterator."""
    def __init__(self, f, stream):
        # only file objects are supported so far
        # check if f is really a file object
        assert isinstance(f, (gzip.GzipFile, file if six.PY2 else io.IOBase))
        self._file_iterator = iter_(f)
        assert isinstance(self._file_iterator, (file_iterator, gfile_iterator))
        self._stream_generator = stream
        self._stream = self._stream_generator(self._file_iterator._f)

    def __next__(self):
        line = self._stream.readline()
        if not line:
            raise StopIteration
        return line

    def __getstate__(self):
        return self._file_iterator, self._stream_generator,\
               self._stream.bytebuffer, self._stream.charbuffer,\
               self._stream.linebuffer, self._stream.errors

    def __setstate__(self, state):
        self._file_iterator, self._stream_generator,\
        bytebuffer, charbuffer, linebuffer, errors = state
        self._stream = self._stream_generator(self._file_iterator._f)
        self._stream.bytebuffer = bytebuffer
        self._stream.charbuffer = charbuffer
        self._stream.linebuffer = linebuffer
        self._stream.errors = errors


class ordered_sequence_iterator(BaseItertool):
    """A picklable replacement for list and tuple iterators."""
    def __init__(self, sequence):
        self._sequence = sequence
        self._position = 0

    def __next__(self):
        if self._position < len(self._sequence):
            value = self._sequence[self._position]
            self._position += 1
            return value
        else:
            raise StopIteration
