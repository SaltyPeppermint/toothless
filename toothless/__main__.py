import sys

import main

update_csv = sys.argv[1] == "--update-csv" if len(sys.argv) >= 2 else False
main.main(update_csv)
