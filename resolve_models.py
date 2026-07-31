#!/usr/bin/env python3
"""查 Ark /models，把 config/models.yaml 里每个逻辑名的最新版本写回 resolved。

供 resolve-models workflow 每天跑：模型一更新/下线，第二天自动落到配置，
你不用手动告诉我具体版本号。
用法：ARK_API_KEY=xxx python resolve_models.py
"""
import sys

from common.logger import log_status, setup_logger
from common.models import resolve_models


def main():
    log = setup_logger()
    try:
        res = resolve_models()  # 读 ARK_API_KEY
        if res.get("ok"):
            log.info("resolve_models 成功: %s", res)
            log_status("resolve-models", True, "ok",
                       resolved=res["resolved"], missing=res.get("missing", []),
                       n_available=res.get("n_available"))
            print("resolved:", res["resolved"])
            if res.get("missing"):
                log.warning("以下逻辑名未匹配到任何版本（可能已下线/改名）: %s", res["missing"])
        else:
            log.error("resolve_models 失败: %s", res.get("err"))
            log_status("resolve-models", False, res.get("err", "失败"))
            sys.exit(1)
    except Exception as e:  # noqa: BLE001
        log.exception("resolve_models 异常")
        log_status("resolve-models", False, str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
