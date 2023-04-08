스크레이핑을 다 하면,
자신의 db파일의 이름을 자기 이름으로 바꾼 뒤(예시)'JW'
여기 Database Folder에다가 옮기고 VS Code상에서 Commit해주세요~!

How to merge databases:
"""
'SID'의 primary key 속성을 제거하고 'SID'를 drop한다
ALTER TABLE table_name DROP COLUMN column_name;

sqlite3에서 JW와 DW을 데이터베이스 연결한 뒤
아래 코드를 입력한다.

CREATE TABLE NaverNewsComments AS
SELECT * FROM projectTable
UNION ALL
SELECT * FROM JW.projectTable
UNION ALL
SELECT * FROM DW.projectTable;
"""


