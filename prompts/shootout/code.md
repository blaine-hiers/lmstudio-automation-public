---
id: code-fizzbuzz-bug
category: code
rubric: Score 1 if no real bug is found or the fix is wrong, 3 if the bug is identified but the fix is incomplete or introduces a new issue, 5 if the exact off-by-one is named and a correct, minimal fix is given.
---
Find and fix the bug in this Python function. Reply with the corrected function only, no explanation.

def fizzbuzz(n):
    for i in range(1, n):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)
