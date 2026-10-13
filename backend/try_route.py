import sys

from app.agent.router import classify

print(classify(" ".join(sys.argv[1:]) or "Why did Cooper leave Murph?"))
