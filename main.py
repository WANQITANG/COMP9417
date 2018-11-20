import sys, getopt
import WNN_prediction
def main(argv):
    # inputfile=''
    # try:
    #     opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    # except getopt.GetoptError:
    #     print('test.py -i <inputfile> ')
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt == '-h':
    #         print('test.py -i <inputfile> ')
    #         sys.exit()
    #     elif opt in ("-i", "--ifile"):
    #         inputfile = arg
    if argv[0]=='WNN_prediction.py':
        return WNN_prediction



if __name__ == "__main__":
    main(sys.argv[1:])