"""AppTest 冒烟验证 AI 基建页能渲染、不报异常。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def test_ai_infra_page_renders():
    """AI 基建页面能渲染且不报异常。"""
    at = AppTest.from_file("quant/web/app.py", default_timeout=30)
    at.run()
    # 切换到 AI 基建页面
    pills = at.pills
    assert len(pills) > 0, "页面导航 pills 不存在"
    # 找到 AI 基建页面
    pills[0].set_value("🤖 AI 基建").run()
    # 检查没有异常
    assert not at.exception, f"AI 基建页面渲染报异常: {at.exception}"


def test_ai_infra_lane_switch():
    """AI 基建切换赛道不报错。"""
    at = AppTest.from_file("quant/web/app.py", default_timeout=30)
    at.run()
    pills = at.pills
    pills[0].set_value("🤖 AI 基建").run()
    assert not at.exception, f"初始渲染报异常: {at.exception}"
    # 尝试切换到另一个赛道
    selectboxes = at.selectbox
    if selectboxes:
        # 找到 ai_infra_lane selectbox
        for sb in selectboxes:
            if sb.key == "ai_infra_lane":
                # 切换到存储赛道
                sb.set_value("存储").run()
                assert not at.exception, f"切换赛道报异常: {at.exception}"
                break


if __name__ == "__main__":
    test_ai_infra_page_renders()
    print("✅ AI 基建页面渲染正常")
    test_ai_infra_lane_switch()
    print("✅ 切换赛道正常")
