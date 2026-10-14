"""Conservative image gate. Brightness is not a scene-safety classifier."""
import io
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import open as decode_image
from ai.core.analysis import ImageQuality

MIN_SIDE = 32
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000
SUPPORTED_FORMATS = {'JPEG', 'PNG', 'WEBP', 'BMP'}


def quality_check(data):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError('image data must be bytes, bytearray or memoryview')
    def reject(issue):
        return None, ImageQuality(usable=False, issues=[issue])
    if not data:
        return reject('EMPTY_IMAGE')
    if len(data) > MAX_BYTES:
        return reject('TOO_LARGE')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            # Preserve the decoder imported before model loading; Ultralytics can
            # replace Image.open with an optional-codec auto-install fallback.
            with decode_image(io.BytesIO(data)) as header:
                if header.format not in SUPPORTED_FORMATS or getattr(header, 'n_frames', 1) != 1:
                    return reject('UNSUPPORTED_FORMAT')
                if header.width * header.height > MAX_PIXELS:
                    return reject('TOO_LARGE')
                if min(header.size) < MIN_SIDE:
                    return reject('TOO_SMALL')
                header.verify()
            with decode_image(io.BytesIO(data)) as original:
                original.load()
                oriented = ImageOps.exif_transpose(original)
                # Ignore hidden RGB values beneath transparency; composite onto black.
                if 'A' in oriented.getbands() or 'transparency' in oriented.info:
                    rgba = oriented.convert('RGBA')
                    if rgba.getchannel('A').getextrema()[1] == 0:
                        return reject('EMPTY_IMAGE')
                    background = Image.new('RGBA', rgba.size, (0, 0, 0, 255))
                    image = Image.alpha_composite(background, rgba).convert('RGB')
                else:
                    image = oriented.convert('RGB')
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return reject('TOO_LARGE')
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, ImportError):
        return reject('DECODE_FAILED')
    low, high = image.convert('L').getextrema()
    # Never reject by average brightness alone: small visible flames matter at night.
    if high <= 2:
        return reject('TOO_DARK')
    if low >= 253:
        return reject('TOO_BRIGHT')
    return image, ImageQuality(usable=True)
