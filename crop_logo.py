from PIL import Image, ImageChops

def autocrop_white(image_path, output_path):
    try:
        im = Image.open(image_path)
        # Handle transparent layers by pasting on white bg
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            alpha = im.convert('RGBA').split()[3]
            bg.paste(im, mask=alpha)
            im = bg
        else:
            im = im.convert('RGB')
            
        bg = Image.new('RGB', im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if bbox:
            w, h = im.size
            # Leave a tight, clean 5px margin
            left = max(0, bbox[0] - 5)
            top = max(0, bbox[1] - 5)
            right = min(w, bbox[2] + 5)
            bottom = min(h, bbox[3] + 5)
            cropped = im.crop((left, top, right, bottom))
            cropped.save(output_path)
            print("Successfully cropped white background")
        else:
            print("No background to crop")
            im.save(output_path)
    except Exception as e:
        print("Error details:", str(e))

autocrop_white("D:/DATN/abc/epc solar.png", "D:/DATN/abc/datn_fe/src/assets/epc_solar_logo.png")
