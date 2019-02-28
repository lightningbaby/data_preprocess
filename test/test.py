import re
a='today is a good day 123 isn is'
b=re.sub('\d','',a)
print(b)