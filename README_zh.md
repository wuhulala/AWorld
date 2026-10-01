# AWorld 1.0

新架构只保留 Agent、Context、Tool，以 Session / Run 提交和管理任务。
源码命名空间、wheel 和命令均为 `aworld`；`aworldv1` 仅为 Lingguang Bench Runtime 内的新 Harness 标识。

```bash
python -m pip install '.[llm,skills]'
aworld run --demo --task hello
aworld run --model MODEL --base-url URL --task '读取最新财报' --skill-path ~/.agents/skills
python -m build
```

API Key 通过 AWORLD_API_KEY / OPENAI_API_KEY 传入。默认 LocalSandbox 提供本地文件和进程，默认工具包含 read、write、bash 以及会话读取、搜索、查询。

旧框架和旧 CLI 已被替换，不做兼容层。默认无第三方依赖，模型连接、Skill 解析分别通过 llm、skills extras 选择；没有构建期间导入业务代码或自动安装依赖。

当前 Session 在进程内保存，Context 使用完整历史，暂未提供跨进程会话持久化、按 token 压缩、流式响应及模型重试策略。

详见 [接口契约](aworld/docs/aworld-1.0-session-run-contract.md) 和 [打包与 CLI](aworld/docs/aworld-1.0-packaging-cli.md)。
