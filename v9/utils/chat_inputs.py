"""Shared image/text chat preparation for online and offline inference."""
import base64
import binascii
import hashlib
import io


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages required')
    images = []
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant'):
            raise ValueError('invalid message role')
        content = message.get('content')
        if isinstance(content, str):
            normalized.append(dict(role=message['role'], content=content))
            continue
        if not isinstance(content, list) or not content:
            raise ValueError('message content must be text or a nonempty list')
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError('invalid content part')
            if part.get('type') == 'text' and isinstance(part.get('text'), str):
                parts.append(dict(type='text', text=part['text']))
            elif part.get('type') == 'image_url':
                image = part.get('image_url')
                if not isinstance(image, dict) or not isinstance(image.get('url'), str):
                    raise ValueError('image_url must contain a URL string')
                images.append(image['url'])
                parts.append(dict(type='image'))
            else:
                raise ValueError('supported content types are text and image_url')
        normalized.append(dict(role=message['role'], content=parts))
    return normalized, images


def decode_image_data(url, max_bytes=8 * 1024**2, max_pixels=16 * 1024**2):
    """Decode a bounded inline image. Network/file URLs are not fetched."""
    from PIL import Image, UnidentifiedImageError
    if not url.startswith('data:image/') or ';base64,' not in url:
        raise ValueError('image_url requires an inline data:image/...;base64 URI')
    payload = url.split(';base64,', 1)[1]
    if len(payload) > 4 * ((max_bytes + 2) // 3):
        raise ValueError('encoded image exceeds byte limit')
    try:
        raw = base64.b64decode(payload, validate=True)
        if len(raw) > max_bytes:
            raise ValueError('image exceeds byte limit')
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > max_pixels:
                raise ValueError('decoded image exceeds pixel limit')
            return image.convert('RGB'), hashlib.sha256(raw).hexdigest()
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('invalid image data') from exc


def prepare_chat_inputs(tokenizer, messages, *, processor=None, max_images=4,
                        max_pixels=12845056, min_pixels=3136):
    normalized, media = validate_messages(messages)
    if not media:
        # Canonicalize text-only content lists for tokenizers without a
        # multimodal template; preserve the usual string-message behavior.
        text_messages = [dict(role=m['role'], content=m['content'] if isinstance(m['content'], str)
                        else ''.join(p['text'] for p in m['content'])) for m in normalized]
        text = tokenizer.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(text, add_special_tokens=False), None
    if processor is None:
        raise ValueError('image input requires multimodal mode')
    if len(media) > max_images:
        raise ValueError('too many images in one request')
    from layers.mm_preprocess import image_to_patches, mrope_positions
    text = processor.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
    enc = processor.tokenizer
    pad = '<|image_pad|>'
    image_id = enc.convert_tokens_to_ids(pad)
    if image_id is None or image_id == enc.unk_token_id:
        raise ValueError('tokenizer has no image placeholder')
    parts = text.split(pad)
    if len(parts) != len(media) + 1:
        raise ValueError('image count disagrees with chat template placeholders')
    ids, images, spans = [], [], []
    for i, url in enumerate(media):
        ids.extend(enc.encode(parts[i], add_special_tokens=False))
        image, source_hash = decode_image_data(url)
        patches, grid = image_to_patches(image, min_pixels=min_pixels, max_pixels=max_pixels)
        t, h, w = grid[0].tolist()
        start = len(ids)
        ids.extend([image_id] * (t * (h // 2) * (w // 2)))
        images.append(dict(pos=start, grid=[t, h, w], pixels=patches.numpy(), source_sha256=source_hash))
        spans.append((start, t, h // 2, w // 2))
    ids.extend(enc.encode(parts[-1], add_special_tokens=False))
    pos3, delta = mrope_positions(len(ids), spans)
    return ids, dict(images=images, pos3=pos3, delta=delta)
