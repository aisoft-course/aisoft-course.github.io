from openai import OpenAI
import os
import subprocess
import json

# =========================
# OpenAI 兼容接口
# =========================
API_KEY = "sk-YOUKEY" #替换为你自己的API_KEY。 


client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

messages = [
    {
        "role": "system",
        "content": """
你是一个CLI智能体。
如果用户请求涉及系统操作（如ls, dir, cat, pwd等），请返回JSON格式：

{
  "action": "run_shell",
  "command": "实际要执行的命令"
}

否则正常回答。
"""
    }
]


def run_shell(cmd: str) -> str:
    """执行shell命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)


def chat(user_input: str):
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )

    content = response.choices[0].message.content.strip()

    # 尝试解析是否是工具调用
    try:
        data = json.loads(content)

        if isinstance(data, dict) and data.get("action") == "run_shell":
            cmd = data["command"]

            print(f"\n[EXECUTING] {cmd}\n")

            output = run_shell(cmd)

            # 把执行结果回传给模型
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"命令执行结果：\n{output}\n请总结结果。"
            })

            final = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                temperature=0.2,
                max_tokens=512,
            )

            reply = final.choices[0].message.content
        else:
            reply = content

    except Exception:
        # 不是JSON就当普通回答
        reply = content

    messages.append({"role": "assistant", "content": reply})
    return reply


def main():
    print("=== CLI Agent with Execution ===")

    while True:
        user_input = input("\nYou: ")

        if user_input in ["exit", "quit"]:
            break

        reply = chat(user_input)
        print("\nAgent:", reply)


if __name__ == "__main__":
    main()
