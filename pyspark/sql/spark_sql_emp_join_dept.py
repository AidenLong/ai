# -- encoding:utf-8 --
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
import time


def parse_emp_record(record):
    """
    处理传入的emp的数组类型的记录信息，转换为Row类型
    :param record:
    :return:
    """
    empno = int(record[0])
    ename = record[1]
    job = record[2]
    try:
        mgr = int(record[3])
    except:
        mgr = -1
    hiredate = record[4]
    sal = float(record[5])
    try:
        comm = float(record[6])
    except:
        comm = -1
    deptno = int(record[7])
    return Row(empno=empno, ename=ename, job=job, mgr=mgr, hiredate=hiredate, sal=sal, comm=comm, deptno=deptno)


# 给定SPARK_HOME环境变量
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'D:\syl\dev\spark-1.6.1-bin-2.5.0-cdh5.3.6'

# 1. 创建上下文
# setMaster: 在开发阶段，必须给定，而且只能是：local、local[*]、local[K]；在集群中运行的时候，是通过命令参数给定，代码中最好注释掉
# setAppName：给定应用名称，必须给定
conf = SparkConf() \
    .setMaster('local') \
    .setAppName('sparksqlreaderdataframe')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sparkContext=sc)

# 2. 读取数据形成RDD
dept_rdd = sc.textFile('../datas/dept.txt') \
    .map(lambda line: line.split('\t')) \
    .filter(lambda arr: len(arr) == 3)
emp_rdd = sc.textFile('../datas/emp.txt') \
    .map(lambda line: line.split('\t')) \
    .filter(lambda arr: len(arr) == 8)

# 3. RDD转换成为DataFrame
dept_row_rdd = dept_rdd.map(lambda arr: Row(deptno=int(arr[0]), dname=arr[1], loc=arr[2]))
emp_row_rdd = emp_rdd.map(lambda arr: parse_emp_record(arr))
dept_df = sql_context.createDataFrame(dept_row_rdd)
emp_df = sql_context.createDataFrame(emp_row_rdd)

# 4. 将DataFrame注册成为临时表
# 临时表的名称中不允许存在符号'.'
dept_df.registerTempTable('tmp_dept')
emp_df.registerTempTable('tmp_emp')
# 如果临时表不再使用了，可以考虑drop掉
# sql_context.dropTempTable('tmp_dept')

# 5. 缓存表数据: 作用是在多次的表数据的获取过程中，加快数据的处理效率
sql_context.cacheTable('tmp_dept')
sql_context.cacheTable('tmp_emp')
# 当缓存的表不需要使用的时候，可以考虑清除缓存
# sql_context.uncacheTable('tmp_dept')

# 6. 业务实现
# 获取sal大于1000的员工名称、部门名称、sal值
sql1 = """
SELECT
  dept.deptno, dept.dname, emp.ename, emp.sal
FROM
  tmp_dept as dept JOIN tmp_emp as emp ON dept.deptno = emp.deptno
WHERE
  emp.sal > 1000
"""
tmp01_df = sql_context.sql(sql1)
tmp01_df.registerTempTable('tmp01')
# 计算各个部门的总的销售额
sql2 = """
SELECT 
  emp.deptno, SUM(sal) as dept_total_sal
FROM 
  tmp_emp as emp
GROUP BY emp.deptno
"""
tmp02_df = sql_context.sql(sql2)
tmp02_df.registerTempTable('tmp02')
# 数据合并
sql3 = """
SELECT
  a.dname, a.ename, a.sal, b.dept_total_sal
FROM
  tmp01 as a JOIN tmp02 as b ON a.deptno = b.deptno
"""
result = sql_context.sql(sql3)

# 7. 结果输出
result.show()
result.write.mode('overwrite').json('../datas/emp_join_dept')

# 暂停一段时间，看一下4040页面
time.sleep(120)
