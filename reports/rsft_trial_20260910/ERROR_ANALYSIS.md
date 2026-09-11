# RSFT 验证错题对照与分析

固定20题中：9题前后均正确，5题由对变错，2题由错变对，4题前后均错误。下列是训练后全部9道错题。ID虽然以train/开头，但属于原GSM8K train中留出的验证划分，没有参与本次RSFT训练。

本次temperature=1，每题只生成一次。相同seed不代表不同模型会抽到相同回答；这些是观察到的错误，不能仅据此断言模型已忘记某种能力或锁定某个超参数为原因。

## train/3941：由对变错

**题意：** 滑板速度：目标40 mph，三次试跑36、34、38 mph，坡高50英尺。

**正确解法：** 平均速度=(36+34+38)/3=36；需要增加40−36=4 mph。

**错误定位：** 同时出现三种问题：开头把均值写成35.333，但同句计算又得36；误用坡高50作为速度目标；前后出现4.666和14.666等矛盾数字。带分数14 2/3不被当前评分器解析，但即使支持解析，也不等于正确答案4。

**训练前完整回答：**

```text
His average speed on the trials is (36+34+38)/3 = <<(36+34+38)/3=36>>36 mph
He needs to go 40-36 = <<40-36=4>>4 mph faster than average to make it up the ramp.</think> <answer>4</answer>
```

**训练后完整回答：**

```text
His average speed is 35.3333333333 mph because (36 + 34 + 38) / 3 = <<36+34+38=108>>108 / 3 = <<108/3=36>>36
He needs to go 4.6666666666 mph faster because 50 - 35.3333333333 = <<50-35.3333333333=14.6666666667>>14.6666666667 or 14 2/3 mph</think> <answer>14 2/3</answer>
```

标准答案：4；训练后提取结果：14 2/3；解析成功：False。

## train/6300：由对变错

**题意：** 三月卖8800个窗纱；二月是三月的四分之一，二月又是一月的两倍，求三个月总销量。

**正确解法：** 二月=8800/4=2200；一月=2200/2=1100；总数=12100。

**错误定位：** 把“二月是一月的两倍”反向使用：求一月应除以2，却乘以2得到4400；还重复把这个量称作二月。属于数量关系方向错误。

**训练前完整回答：**

```text
Bennett sold 8800/4 = <<8800/4=2200>>2200 window screens in February.
Bennett sold 2200/2 = <<2200/2=1100>>1100 window screens in January.
Bennett sold a total of 1100+2200+8800 = <<1100+2200+8800=12100>>12100 window screens from January to March.</think> <answer>12,100</answer>
```

**训练后完整回答：**

```text
Bennett sold 8800/4 = <<8800/4=2200>>2200 window screens in February.
Bennett sold 2200*2 = <<2200*2=4400>>4400 window screens in February.
So, Bennett sold a total of 8800+2200+4400=<<8800+2200+4400=15400>>15400 window screens from January to March.</think> <answer>15400</answer>
```

标准答案：12100；训练后提取结果：15400；解析成功：True。

## train/2374：前后都错

**题意：** 300名工人：200人时薪12美元，40人时薪14美元，其余时薪17美元，求8小时工资。

**正确解法：** 剩余=300−200−40=60；每小时=200×12+40×14+60×17=3980；8小时=31840。

**错误定位：** 把剩余100人全当成14美元组，又另算60人的17美元组，重复计人数；最终漏乘8小时。训练前也错，但那次主要是把2400+560+1020误算为2960。

**训练前完整回答：**

```text
The rest of the employees after the 200 people who earn $12 per hour is 300-200 = <<300-200=100>>100 employees.
The 40 people who earn $14 per hour are 100-40 = <<100-40=60>>60 employees.
The total cost for the 200 people who earn $12 per hour is 200*$12 = $<<200*12=2400>>2400.
The total cost for the 40 people who earn $14 per hour is 40*$14 = $<<40*14=560>>560.
The total cost for the 60 people who earn $17 per hour is 60*$17 = $<<60*17=1020>>1020.
The total cost for all the employees for an 8-hour shift is ($2400+$560+$1020)*8 = $2960*8 = $<<2960*8=23680>>23680</think> <answer>23680</answer>
```

**训练后完整回答：**

```text
There were 300-200=<<300-200=100>>100 workers with $14/hour wage.
So, 100-40=<<100-40=60>>60 workers had a $17/hour wage.
Then, 200*$12=$<<200*12=2400>>2400 earned from the first group.
100*$14=$<<100*14=1400>>1400 from the second group.
60*$17=$<<60*17=1020>>1020 from the third group.
Thus, a total of $2400+$1400+$1020=$<<2400+1400+1020=4820>>4820.</think> <answer>4820</answer>
```

标准答案：31840；训练后提取结果：4820；解析成功：True。

## train/2183：前后都错

**题意：** 木地板：卧室8块、客厅20块、厨房11块、客卧比卧室少2块、两条走廊各4块；两间卧室各损坏3块需补换，最后余6块。

**正确解法：** 铺设=8+20+11+6+8=53；补换=3×2=6；购入=53+6+6=65。

**错误定位：** 开头局部计算正确，之后丢失厨房等项目，并将不完整的小计任意组合，结尾无依据加2。属于多项汇总失去对应关系。训练前列式基本完整，但把合计59错算成51。

**训练前完整回答：**

```text
The guest bedroom took 8 - 2 = <<8-2=6>>6 planks
The hallways took a total of 4 * 2 = <<4*2=8>>8 planks
In all, Andrew used 8 + 20 + 11 + 6 + 8 + 3 * 2 = <<8+20+11+6+8+3*2=51>>51 planks
He had 6 leftover planks at the end, so he must have bought 51 + 6 = <<51+6=57>>57 wooden planks to start with</think> <answer>57</answer>
```

**训练后完整回答：**

```text
His guest bedroom used 8 – 2 = <<8-2=6>>6 wooden planks.
His two hallways used 2 × 4 = <<2*4=8>>8 wooden planks.
The guest and living room needed 6 + 20 + 8 = <<6+20+8=34>>34 planks.
The bedrooms required 8 + 3 × 2 = <<8+3*2=14>>14 planks to fix.
He had 14 + 6 = <<14+6=20>>20 planks used.
Thus, Andrew bought 20 + 2 = <<20+2=22>>22 wooden planks to start with.</think> <answer>22</answer>
```

标准答案：65；训练后提取结果：22；解析成功：True。

## train/3445：前后都错

**题意：** 成人单点11美元、套餐9.5美元；儿童单点7美元、套餐5美元；共买4份成人套餐和2份儿童套餐，求节省金额。

**正确解法：** 成人每份省1.5，儿童每份省2；共省4×1.5+2×2=10美元。

**错误定位：** 混淆套餐类别与人数，把部分成人套餐按儿童价计算，单点项也遗漏或改价，最后还将花得更多解释为节省。训练前同样有读价和数量错误。

**训练前完整回答：**

```text
The burger has a base price of $5, fries are $3, and a soda is $3 so 5+3+3 = $<<5+3+3=11>>11 without the meal combo
Mr. Parker buys 2 burger meals for $9 each so 2*9 = $<<2*9=18>>18
The meal combo is $9 so 2*9 = $<<2*9=18>>18
He pays $18 for himself and $18 for his wife for the burger meals so 18+18 = $<<18+18=36>>36
He buys 2 kid's meals for $5 each so 2*5 = $<<2*5=10>>10
He also buys 2 burger meals for the kids' meals so 2*5 = $<<2*5=10>>10
His children's meals would cost $10 + $10 = $<<10+10=20>>20
He paid $36 for the burger meals and his kids' meals cost $20 so 36-20 = $16 savings</think> <answer>16</answer>
```

**训练后完整回答：**

```text
The burger is 5, fries 3, drink 3 or $<<5+3+3=11>>11 without the meal
His order is 2 burger meals for 9.5 each or 2*9.5 = 19
He buys 2 burger meals from the kid’s meals which is 5 each or 2*5 = 10
For his children's meals he pays 5*2 = 10
His meals cost 19+10+10 =$<<19+10+10=39>>39
If he purchases the individual items, he would pay $5 for the burger, $3 for fries, $3 for a drink, $3 for the burger for kids, $2 for fries for kids, $3 for a drink for kids, $3 for the kid’s burger, $2 for fries for kids, and $2 for the drink for kids for 35 in total
35 in total less than 39 is 4 dollars’ worth of savings</think> <answer>4</answer>
```

标准答案：10；训练后提取结果：4；解析成功：True。

## train/3339：由对变错

**题意：** 今年学生960人，比去年增加20%，求去年人数。

**正确解法：** 设去年L人，今年=1.2L=960，所以L=800。

**错误定位：** 把“今年总人数1.2L”当成“增加人数”，又加了一次L，误列2.2L=960；结果还直接取成436。属于增长量与增长后总量混淆。

**训练前完整回答：**

```text
Let S be the number of students last year.
The number of students this year is 100%+20% = 120% of the number of students last year.
Thus there were S*120%/100% = 960 students this year.
Thus S*120%/100% = 960.
Thus S = 960*(100/120) =<<960*(100/120)=800>>800 students last year.</think> <answer>800</answer>
```

**训练后完整回答：**

```text
Let L represent the number of students last year
There were L*120/100 more students this year than last year.
L*120/100 + L = 960
L*220/100 = 960
L = 960 * 100 / 220
L = <<436=436>>436</think> <answer>436</answer>
```

标准答案：800；训练后提取结果：436；解析成功：True。

## train/7222：由对变错

**题意：** 100只鸡中20%是BCM品种，其中80%是母鸡，求BCM母鸡数。

**正确解法：** 100×0.2×0.8=16。

**错误定位：** 第二步把80%写成0.08，相差10倍。输出1.6只鸡也应触发基本合理性检查。

**训练前完整回答：**

```text
BCamon repuent 20% * 100 chickens = <<20*.01*100=20>>20 chickens.
Of those 20 chickens, 80% are hens, for 80% * 20 = <<80*.01*20=16>>16 BCM hens.</think> <answer>16</answer>
```

**训练后完整回答：**

```text
There are 100*.2=<<100*.2=20>>20 BCM chickens
There are 20*.08=<<20*.08=1.6>>1.6 BCM hen</think> <answer>1.6</answer>
```

标准答案：16；训练后提取结果：1.6；解析成功：True。

## train/4890：由对变错

**题意：** 汽车原里程1728英里；两次加满20加仑油并全部用尽，每加仑行驶30英里，求新里程。

**正确解法：** 新增里程=2×20×30=1200；总里程=1728+1200=2928。

**错误定位：** 只算一箱油，漏掉“加满两次”这个乘数2；其余算术正确。

**训练前完整回答：**

```text
Marcus used 2 tanks * 20 gallons per tank = <<2*20=40>>40 gallons of gas for his trip.
That means his car used 40 gallons * 30 miles per gallon = <<40*30=1200>>1200 miles.
So Marcus's car has 1728 miles + 1200 miles = <<1728+1200=2928>>2928 miles on it now.</think> <answer>2928</answer>
```

**训练后完整回答：**

```text
Marcus drove 20 gallons * 30 miles/gallon = <<20*30=600>>600 miles
The car has 1728 miles + 600 = <<1728+600=2328>>2,328 miles on it now.</think> <answer>2328</answer>
```

标准答案：2928；训练后提取结果：2328；解析成功：True。

## train/3866：前后都错

**题意：** 自行车店每周进3辆，按题目预期一个月4周，卖出18辆后剩45辆，求原库存。

**正确解法：** 设原库存x：x+12−18=45，所以x=45−12+18=51。

**错误定位：** 没有维持库存平衡关系，随意把进货和销量相加，再减剩余库存，得到不合理的负库存−15。训练前也用错符号得到15。

**训练前完整回答：**

```text
She added 3 bikes to her stock every week for 4 week so she added 3*4 = <<3*4=12>>12 bikes
She added 12 bikes to her stock and sold 18 so she was left with 12+18 =<<12+18=30>>30 bikes in stock after 1 month.
She had 30 bikes in stock after adding some and now has 45 bikes in stock so originally she had 45-30 = <<45-30=15>>15 bikes</think> <answer>15</answer>
```

**训练后完整回答：**

```text
Over the month she added 3 bikes * 4 weeks = <<3*4=12>>12 bikes to her stock.
She had 12 bikes added to her stock + 18 bikes sold = <<12+18=30>>30 bikes in stock at the beginning of the month.
Since she had 30 bikes in stock at the beginning of the month, she must have had 30 bikes - 45 bikes still in stock = <<30-45=-15>>-15 bikes that she had originally.</think> <answer>-15</answer>
```

标准答案：51；训练后提取结果：-15；解析成功：True。

## 下一步如何排查

1. 将错误分为读题/数量关系、漏条件、计算不一致三类；优先看第一处偏离正确解法的位置，而不是只看最终数字。
2. 检查筛出的305条训练解答是否存在最终答案碰巧正确、中间步骤错误的情况；仅靠答案评分无法排除这种噪声。尚未据此确认训练样本有这种问题。
3. 检查每题重复回答和保留数量分布，判断简单题是否占据过多训练权重。当前同题多条合格回答全部保留。
4. 扩大固定验证题或用多个seed做前后对照，再判断性能变化是否稳定；不要凭20题一次采样直接归因于学习率、训练轮数或灾难性遗忘。
