import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

import img2pptx as core


class PictureTextTests(unittest.TestCase):
    def convert(self, photograph, alpha_mode="solid", external_label=False, text_edits=None):
        image = Image.new('RGB', (500, 300), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 450, 260), fill=(230, 235, 240))
        font = core.ref_font(36)
        draw.text((130, 150), 'REPORT', fill=(10, 35, 85), font=font, anchor='ls')
        lines = [core.Line('REPORT', 128, 115, 158, 39, score=0.99)]
        if external_label:
            draw.text((456, 150), 'OK', fill=(10, 35, 85), font=core.ref_font(20), anchor='ls')
            lines.append(core.Line('OK', 454, 130, 36, 25, score=0.99))
        bgr = np.asarray(image)[:, :, ::-1].copy()
        box = (40, 40, 410, 220)
        sel = np.zeros((300, 500), bool)
        sel[40:260, 40:450] = True
        alpha = np.full((220, 410), 255, np.uint8)
        if alpha_mode == "text_holes":
            glyph_mask = Image.new('L', (500, 300), 0)
            ImageDraw.Draw(glyph_mask).text((130, 150), 'REPORT', fill=255, font=font, anchor='ls')
            alpha[np.asarray(glyph_mask)[40:260, 40:450] > 0] = 0
        elif alpha_mode == "ring":
            alpha.fill(0)
            cv2.rectangle(alpha, (0, 0), (409, 219), 255, thickness=10)
        rgba = np.dstack((bgr[40:260, 40:450], alpha))
        picture = core.Icon(40, 40, rgba, '图片', box, 90, 'fixture')
        zeros = np.zeros((300, 500), np.uint8)
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            path = Path(tmp) / 'source.png'
            cv2.imwrite(str(path), bgr)
            stack.enter_context(patch.object(core, 'extract_pictures', return_value=(
                [picture], sel.astype(np.uint8)*255, [sel], zeros.copy())))
            stack.enter_context(patch.object(core, 'find_photo_blocks', return_value=[box] if photograph else []))
            stack.enter_context(patch.object(core, 'find_picture_regions', return_value=[]))
            stack.enter_context(patch.object(core, 'find_illustrations', return_value=[]))
            stack.enter_context(patch.object(core, 'find_leftover_pictures', return_value=[]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.source = bgr
            return core.process_image(str(path), 'rapid', 'eng', with_icons=False,
                                      inpaint='telea', ocr=lines, text_edits=text_edits)

    def test_existing_photo_keeps_internal_text_in_picture(self):
        _, blocks, pictures, _, _ = self.convert(photograph=True)
        self.assertEqual(blocks, [])
        self.assertEqual(len(pictures), 1)

    def test_internal_text_stays_in_opaque_illustration(self):
        _, blocks, pictures, _, _ = self.convert(photograph=False, alpha_mode='text_holes')
        self.assertEqual(blocks, [])
        self.assertEqual(len(pictures), 1)
        crop = pictures[0].rgba[75:115, 88:246]
        source = self.source[115:155, 128:286]
        glyphs = np.max(np.abs(source.astype(np.int16) - np.array([240, 235, 230])), axis=2) > 80
        self.assertGreater(int(glyphs.sum()), 0)
        np.testing.assert_array_equal(crop[..., 3][glyphs], np.full(int(glyphs.sum()), 255, np.uint8))
        np.testing.assert_array_equal(crop[..., :3][glyphs], source[glyphs])

    def test_adjacent_external_label_stays_editable(self):
        _, blocks, _, _, _ = self.convert(photograph=False, external_label=True)
        self.assertEqual([line.text for block in blocks for line in block.lines], ['OK'])

    def test_hollow_component_does_not_hide_internal_text(self):
        _, blocks, _, _, _ = self.convert(photograph=False, alpha_mode='ring')
        self.assertEqual([line.text for block in blocks for line in block.lines], ['REPORT'])

    def test_manual_text_stays_editable_inside_opaque_illustration(self):
        edits = [{'box': [120, 110, 180, 50], 'text': 'MANUAL'}]
        _, blocks, _, _, _ = self.convert(photograph=False, text_edits=edits)
        self.assertIn('MANUAL', [line.text for block in blocks for line in block.lines])


if __name__ == '__main__':
    unittest.main()
