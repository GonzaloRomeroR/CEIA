# Connect to the database
import pymysql

connection = pymysql.connect(
    host="34.133.210.185",
    user="root",
    password="1234",
    cursorclass=pymysql.cursors.DictCursor,
)
