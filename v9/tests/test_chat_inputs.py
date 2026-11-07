"""Request boundary tests: preserve images/order and reject malformed media."""
import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from utils.chat_inputs import prepare_chat_inputs, decode_image_data


class Tokenizer:
    unk_token_id = -1
    def encode(self, text, **kwargs): return list(text.encode())
    def convert_tokens_to_ids(self, text): return 1000
    def apply_chat_template(self, messages, **kwargs):
        return '|'.join(m['content'] for m in messages)


def uri(color):
    b = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(b, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def test_image_order_and_text_only_fallback():
    tok = Tokenizer()
    text_ids, mm = prepare_chat_inputs(tok, [{'role':'user','content':[{'type':'text','text':'ab'}]}])
    assert text_ids == [97,98] and mm is None
    processor = SimpleNamespace(tokenizer=tok, apply_chat_template=lambda *a, **k:
                                'a<|image_pad|>b<|image_pad|>c')
    parts = [{'type':'image_url','image_url':{'url':uri(c)}} for c in ('red','blue')]
    ids, mm = prepare_chat_inputs(tok, [{'role':'user','content':parts}], processor=processor)
    assert ids == [97]+[1000]*4+[98]+[1000]*4+[99]
    assert [im['pos'] for im in mm['images']] == [1,6]
    assert mm['pos3'].shape == (3,len(ids))
    assert mm['delta'] == int(mm['pos3'].max())+1-len(ids)
    assert not np.array_equal(mm['images'][0]['pixels'],mm['images'][1]['pixels'])


@pytest.mark.parametrize('url',['https://example.com/x.png','file:///a','data:image/png;base64,!!!',
                              'data:image/png;base64,YWJj'])
def test_invalid_media_is_rejected(url):
    with pytest.raises(ValueError): decode_image_data(url)


def test_image_limit_applies_before_pixel_decode():
    with pytest.raises(ValueError, match='byte limit'):
        decode_image_data(uri('red'), max_bytes=2)
    with pytest.raises(ValueError, match='pixel limit'):
        decode_image_data(uri('red'), max_pixels=8)
