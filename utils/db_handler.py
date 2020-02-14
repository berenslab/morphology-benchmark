import pandas as pd
import psycopg2 as ps
import numpy as np

from io import StringIO


def connect():
    conn = ps.connect("dbname='morphologies' user='slaturnus' host='localhost' password='morphologies'")
    conn.set_session(autocommit=True)
    return conn


# loads data from the morphology data base.
def load_data_from_db(conn, data_type, condition=None):
    # I know this is horrible style since that is code injection by itself
    sql = """SELECT cells.id, internal_id, cell_type, data FROM cells JOIN data on cells.id = cell_id
            WHERE data_type = (%(dt)s);"""
    sql_condition = """SELECT cells.id, internal_id, cell_type, data FROM cells JOIN data on cells.id = cell_id WHERE
                    data_type = (%(dt)s) AND {0};""".format(condition)

    cur = conn.cursor()
    d = {'dt': data_type}

    if condition:
        cur.execute(sql_condition, d)
    else:
        cur.execute(sql, d)

    r = cur.fetchall()
    r_array = np.array(r)

    result = dict()
    result['cell_id'] = [int(x) for x in r_array[:, 0]]
    result['internal_id'] = [int(x) for x in r_array[:, 1]]
    result['cell_type'] = r_array[:, 2]
    result['data'] = r_array[:, 3]
    data = []
    if data_type == 'swc':
        # get swc in appropriate format
        for ii in range(len(r)):
            swc = pd.read_csv(StringIO(r[ii][3]))
            data.append(swc)
        result['data'] = data
    elif data_type in ['asc', 'dat']:
        print('This method is not implemented yet')
    else:
        print('Unknown data type. Data is returned in raw format')

    return result
