"""推送通道：把 scan 信号推到企微/飞书/Server酱。

支持通道：
- wecom_robot:  企业微信群机器人 webhook（推荐起步）
- feishu_robot: 飞书自定义机器人 webhook
- serverchan:   Server酱·Turbo（推到个人微信）

配置：config/notify.yaml
  channels:
    - type: wecom_robot
      webhook: https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
      enabled: true
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import requests
import yaml

from qkquant.logger import logger


class Notifier(Protocol):
    def send(self, title: str, content: str) -> bool: ...


class WecomRobot:
    """企业微信群机器人。markdown 内容上限 4096 字符。"""

    LIMIT = 4096

    def __init__(self, webhook: str) -> None:
        self.webhook = webhook

    def send(self, title: str, content: str) -> bool:
        text = f"# {title}\n\n{content}"
        if len(text) > self.LIMIT:
            text = text[: self.LIMIT - 60] + "\n\n... (truncated)"
        try:
            r = requests.post(
                self.webhook,
                json={"msgtype": "markdown", "markdown": {"content": text}},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("errcode") != 0:
                logger.error(f"wecom send failed: {data}")
                return False
            return True
        except Exception as e:
            logger.error(f"wecom send error: {e}")
            return False


class FeishuRobot:
    """飞书自定义机器人。"""

    def __init__(self, webhook: str) -> None:
        self.webhook = webhook

    def send(self, title: str, content: str) -> bool:
        text = f"{title}\n\n{content}"
        try:
            r = requests.post(
                self.webhook,
                json={"msg_type": "text", "content": {"text": text}},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("code") not in (0, None) and data.get("StatusCode") not in (0, None):
                logger.error(f"feishu send failed: {data}")
                return False
            return True
        except Exception as e:
            logger.error(f"feishu send error: {e}")
            return False


class DingtalkRobot:
    """钉钉自定义机器人。markdown 上限 5000 字符。

    安全设置三选一（钉钉群机器人配置时选）:
      a) 自定义关键词: 消息内容必须包含设定的关键词（最简单, 此实现自动把 title 加在开头）
      b) 加签 (HMAC-SHA256): 设置 secret 后会自动签名 URL
      c) IP 白名单: 服务器固定 IP 时使用

    yaml 配置:
        type: dingtalk_robot
        enabled: true
        webhook: "https://oapi.dingtalk.com/robot/send?access_token=xxx"
        secret: "SECxxxxxxxxxxxxx"     # 可选; 加签模式才填
        keyword: "qkquant"              # 可选; 关键词模式留个标记
    """

    LIMIT = 5000

    def __init__(self, webhook: str, secret: str | None = None, keyword: str | None = None) -> None:
        self.webhook = webhook
        self.secret = secret
        self.keyword = keyword

    def _signed_url(self) -> str:
        if not self.secret:
            return self.webhook
        import base64
        import hashlib
        import hmac
        import time
        from urllib.parse import quote_plus

        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}".encode("utf-8")
        hmac_code = hmac.new(
            self.secret.encode("utf-8"), string_to_sign, digestmod=hashlib.sha256
        ).digest()
        sign = quote_plus(base64.b64encode(hmac_code))
        sep = "&" if "?" in self.webhook else "?"
        return f"{self.webhook}{sep}timestamp={timestamp}&sign={sign}"

    def send(self, title: str, content: str) -> bool:
        # 关键词模式: 把 keyword 拼到 title 里以确保通过
        if self.keyword and self.keyword not in title:
            title = f"{self.keyword} {title}"
        text = f"# {title}\n\n{content}"
        if len(text) > self.LIMIT:
            text = text[: self.LIMIT - 60] + "\n\n... (truncated)"
        try:
            r = requests.post(
                self._signed_url(),
                json={
                    "msgtype": "markdown",
                    "markdown": {"title": title, "text": text},
                    "at": {"isAtAll": False},
                },
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("errcode") != 0:
                logger.error(f"dingtalk send failed: {data}")
                return False
            return True
        except Exception as e:
            logger.error(f"dingtalk send error: {e}")
            return False


class ServerChan:
    """Server酱·Turbo: 推到个人微信（单聊）。"""

    def __init__(self, sendkey: str) -> None:
        self.sendkey = sendkey

    def send(self, title: str, content: str) -> bool:
        try:
            r = requests.post(
                f"https://sctapi.ftqq.com/{self.sendkey}.send",
                data={"title": title, "desp": content},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("code") != 0:
                logger.error(f"serverchan send failed: {data}")
                return False
            return True
        except Exception as e:
            logger.error(f"serverchan send error: {e}")
            return False


def load_notifiers(path: Path) -> list[Notifier]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    notifiers: list[Notifier] = []
    for ch in cfg.get("channels", []):
        if not ch.get("enabled", True):
            continue
        ctype = ch.get("type")
        try:
            if ctype == "wecom_robot":
                notifiers.append(WecomRobot(ch["webhook"]))
            elif ctype == "feishu_robot":
                notifiers.append(FeishuRobot(ch["webhook"]))
            elif ctype == "dingtalk_robot":
                notifiers.append(
                    DingtalkRobot(
                        ch["webhook"],
                        secret=ch.get("secret") or None,
                        keyword=ch.get("keyword") or None,
                    )
                )
            elif ctype == "serverchan":
                notifiers.append(ServerChan(ch["sendkey"]))
            else:
                logger.warning(f"unknown notifier type: {ctype}")
        except KeyError as e:
            logger.error(f"notify config missing key for {ctype}: {e}")
    return notifiers


def push_all(notifiers: list[Notifier], title: str, content: str) -> int:
    n_ok = 0
    for n in notifiers:
        if n.send(title, content):
            n_ok += 1
    return n_ok


__all__ = [
    "Notifier",
    "WecomRobot",
    "FeishuRobot",
    "DingtalkRobot",
    "ServerChan",
    "load_notifiers",
    "push_all",
]
