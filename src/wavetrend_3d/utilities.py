# utilities.py
# Generally applicable handy functions
# Copyright (C) 2024 Arthur Nazarian


def hex_to_rgba(hex_color: str, opacity: float = 1.0, offset: int = 0, return_type: str = 'string') -> (str | tuple):
    """
    Convert a hex color to RGBA.

    Parameters
    ----------
    hex_color : str
       The hexadecimal color to convert. It should be in the format "#RRGGBB" or "RRGGBB".
    opacity : float, optional
       The opacity level. It should be in the range [0.0, 1.0]. Default is 1.0 (fully opaque).
    offset : int, optional
        An offset to adjust the color's brightness. Positive values lighten the color, while negative values
        darken it. Default is 0 (no adjustment).
    return_type : string, optional
        Either 'string' (default) or 'tuple'. The former returns a string like `"rgba(255, 100, 40, 0.2)"`
        that can be used in e.g. Plotly. The latter returns a tuple in the form of `(255, 100, 40, 0.2).

    Return
    -------
    str | tuple
       A string or tuple in the format `"rgba(r, g, b, a)"` or `(r, g, b, a)` respectively, where `r`, `g`, `b` are the
       RGB (red-green-blue) values (range 0-255) and `a` is the opacity (range 0-1).

   """
    if not isinstance(hex_color, str):
        raise ValueError("The hex color should be a string.")
    if not isinstance(opacity, (float, int)) or not 0.0 <= opacity <= 1.0:
        raise ValueError("The opacity should be a float in the range [0.0, 1.0].")
    if not isinstance(offset, int) or not -255 <= offset <= 255:
        raise ValueError("The offset should be an integer in the range [-255, 255].")

    if hex_color.startswith("#"):
        hex_color = hex_color[1:]

    if len(hex_color) != 6:
        raise ValueError("The hex color should be in the format 'RRGGBB' or '#RRGGBB'.")

    try:
        r = max(0, min(255, int(hex_color[0:2], 16) + offset))
        g = max(0, min(255, int(hex_color[2:4], 16) + offset))
        b = max(0, min(255, int(hex_color[4:6], 16) + offset))
    except ValueError:
        raise ValueError("The hex color should only contain hexadecimal digits.")

    if return_type == 'string':
        return f"rgba({r}, {g}, {b}, {opacity})"
    elif return_type == 'tuple':
        return r, g, b, opacity
    else:
        raise ValueError(f'`return_type` should be either "string" or "tuple"; received {return_type}')
