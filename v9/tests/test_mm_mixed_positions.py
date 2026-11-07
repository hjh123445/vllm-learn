"""A mixed prefill batch must retain the decode positions of image requests."""
import numpy as np
from layers.mm_preprocess import mrope_positions, mrope_position_window


def test_window_crossing_prompt_end_keeps_delta_and_all_rows():
    prompt, delta = mrope_positions(13, [(2, 1, 2, 4)])
    actual = mrope_position_window(prompt, 12, 4, delta)
    expected = np.concatenate([prompt[:,12:13], np.tile(np.arange(13,16)+delta,(3,1))],axis=1)
    np.testing.assert_array_equal(actual,expected)


def test_mixed_batch_generated_decode_and_new_image_prefill():
    old, delta = mrope_positions(13, [(2, 1, 2, 4)])
    new, _ = mrope_positions(9, [(2, 1, 2, 2)])
    # The earlier request is two generated tokens past its prompt, while a
    # newly admitted request starts prefill in the same flat attention batch.
    rows = np.concatenate([mrope_position_window(old,15,1,delta),
                           mrope_position_window(new,0,9)],axis=1)
    assert rows.shape == (3,10)
    np.testing.assert_array_equal(rows[:,0],np.full(3,15+delta))
    np.testing.assert_array_equal(rows[:,1:],new)


def test_decode_only_metadata_uses_delta_without_prompt_table():
    np.testing.assert_array_equal(mrope_position_window(None,30,2,-7),[[23,24]]*3)
