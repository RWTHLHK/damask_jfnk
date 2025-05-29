import os
import time
import requests
import re

SCKEY = "SCT281072T4BhUWJ5NlLAbDaUNFCddg8sb"  # <-- 在此处填写你的SCKEY
STA_FILE = "/home/doelz-admin/projects/damask_jfnk/workdir/grid_load_material.sta"
CHECK_INTERVAL = 60  # seconds

def send_wechat(msg):
    url = f"https://sctapi.ftqq.com/{SCKEY}.send"
    data = {"title": "DAMASK模拟进度更新", "desp": msg}
    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            print("[通知] 已推送到微信！")
        else:
            print(f"[通知] 推送失败，状态码: {resp.status_code}")
    except Exception as e:
        print(f"[通知] 推送异常: {e}")

def get_last_increment(sta_file):
    if not os.path.exists(sta_file):
        return None
    try:
        with open(sta_file, "r") as f:
            lines = f.readlines()
        # 假设首行为表头或空行，真实 increment 数量为 len(lines) - 1
        return len(lines) - 1 if len(lines) > 1 else 0
    except Exception as e:
        print(f"[错误] 读取 {sta_file} 失败: {e}")
        return None

if __name__ == "__main__":
    print(f"监控 {STA_FILE}，每{CHECK_INTERVAL}s检查一次 increment 是否有更新...")
    last_seen_increment = get_last_increment(STA_FILE)
    if last_seen_increment is not None:
        print(f"初始 increment: {last_seen_increment}")
    else:
        print("未检测到初始 increment，等待文件生成...")

    send_wechat(f"测试消息：DAMASK监控推送功能正常！当前increment= {last_seen_increment}")

    while True:
        time.sleep(CHECK_INTERVAL)
        current_increment = get_last_increment(STA_FILE)
        if current_increment > last_seen_increment:
            msg = f"grid_load_material.sta 检测到新 increment: {current_increment} (上次: {last_seen_increment})"
            print(f"[监控] {msg}")
            send_wechat(msg)
            last_seen_increment = current_increment

