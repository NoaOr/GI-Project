# import pandas as pd
#
# from sqlalchemy import create_engine
#
#
# if __name__ == '__main__':
#     t1 = pd.read_excel('table1.xls')
#     df1 = pd.DataFrame(t1)
#     a1 = df1.ix[0]
#     print(a1)
#     print("\n\n")
#     t2 = pd.read_excel('table2.xls')
#     df2 = pd.DataFrame(t2)
#     a2 = df2.ix[0]
#     print(a2)
#
#     #
#     # engine = create_engine('sqlite:///:memory:')
#     # pd.read_sql("SELECT * FROM my_table;", engine)
#     # pd.read_sql_table('my_table', engine)
#     # pd.read_sql_query("SELECT * FROM my_table;", engine)