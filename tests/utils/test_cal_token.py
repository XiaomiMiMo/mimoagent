import pytest

from mimoagent.utils.cal_token import MEDIA_BLOCK_TOKEN_ESTIMATE, rough_token_count_estimation_for_message


def test_native_reasoning_encrypted_content_counts_toward_history_size():
    item = {"type": "reasoning", "encrypted_content": "x" * 400}

    assert rough_token_count_estimation_for_message(item) == 100


def test_native_reasoning_without_encrypted_content_stays_zero():
    item = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "not counted"}]}

    assert rough_token_count_estimation_for_message(item) == 0


@pytest.mark.parametrize("payload_size", [4, 1_398_104])
@pytest.mark.parametrize("role", ["user", "tool"])
def test_image_urls_count_as_images_instead_of_base64_text(role, payload_size):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * payload_size}}
    message = {"role": role, "content": [{"type": "text", "text": "abcdefgh"}, image, image]}

    assert rough_token_count_estimation_for_message(message) == 2 + 2 * MEDIA_BLOCK_TOKEN_ESTIMATE
