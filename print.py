import asyncio
import logging
import sys
import os

import click

from catprinter.cmds import PRINT_WIDTH, cmds_print_img
from catprinter.ble import run_ble
from catprinter.img import read_img, text_to_img

def make_logger(log_level):
    logger = logging.getLogger('catprinter')
    logger.setLevel(log_level)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(log_level)
    logger.addHandler(h)
    return logger

@click.group()
def cli():
    pass

def print_img(
    img,
    devicename,
    darker,
    logger
):
    logger.info(f'✅ Read image: {img.shape} (h, w) pixels')
    data = cmds_print_img(img, dark_mode=darker)
    logger.info(f'✅ Generated BLE commands: {len(data)} bytes')

    # Try to autodiscover a printer if --devicename is not specified.
    autodiscover = not devicename
    return asyncio.run(run_ble(data, devicename, autodiscover, logger))

@click.command()
@click.option("--filename", type=str, required=True)
@click.option('--log-level', 
    type=click.Choice(['debug', 'info', 'warn', 'error'], case_sensitive=False), default="info")  # TODO generalise for all functions
@click.option('--img-binarization-algo', 
    type=click.Choice(['mean-threshold', 'floyd-steinberg', 'halftone', 'none'], case_sensitive=False),
    default='floyd-steinberg',
    help=f'Which image binarization algorithm to use. If \'none\' is used, no binarization will be used. In this case the image has to have a width of {PRINT_WIDTH} px.'
)
@click.option('--show-preview/--skip-preview', default=False)
@click.option("--devicename", type=str,  
    help='Specify the Bluetooth Low Energy (BLE) device name to    \
        search for. If not specified, the script will try to       \
        auto discover the printer based on its advertised BLE      \
        service UUIDs. Common names are similar to "GT01", "GB02", \
        "GB03".'
)
@click.option('--darker/--not-darker', default=False, 
     help="Print the image in text mode. This leads to more contrast, but slower speed.")
def image(filename, log_level, img_binarization_algo, show_preview, devicename, darker):
    log_level = getattr(logging, log_level.upper())
    logger = make_logger(log_level)

    if not os.path.exists(filename):
        logger.info('🛑 File not found. Exiting.')
        return

    bin_img = read_img(filename, PRINT_WIDTH,
                       logger, img_binarization_algo, show_preview)
    if bin_img is None:
        logger.info(f'🛑 No image generated. Exiting.')
        return

    return print_img(
        img=bin_img,
        devicename=devicename,
        darker=darker,
        logger=logger
    )

@click.command()
@click.option("--text", type=str, required=True)
@click.option("--font", type=str, default="Verdana Bold.ttf")
@click.option("--font-size", type=int, default=30)
@click.option('--log-level', 
    type=click.Choice(['debug', 'info', 'warn', 'error'], case_sensitive=False), default="info")  # TODO generalise for all functions
@click.option('--show-preview/--skip-preview', default=False)
@click.option("--devicename", type=str,  
    help='Specify the Bluetooth Low Energy (BLE) device name to    \
        search for. If not specified, the script will try to       \
        auto discover the printer based on its advertised BLE      \
        service UUIDs. Common names are similar to "GT01", "GB02", \
        "GB03".'
)
def text(text, font, font_size, log_level, show_preview, devicename):
    log_level = getattr(logging, log_level.upper())
    logger = make_logger(log_level)

    bin_img = text_to_img(
        text=text.replace("\\n", "\n"),
        font_name=font,
        font_size=font_size,
        show_preview=show_preview,
        logger=logger
    )
    if bin_img is None:
        logger.info(f'🛑 No text image generated. Exiting.')
        return

    return print_img(
        img=bin_img,
        devicename=devicename,
        darker=True,  # Text mode
        logger=logger
    )

if __name__ == '__main__':
    cli.add_command(image)
    cli.add_command(text)
    cli()
