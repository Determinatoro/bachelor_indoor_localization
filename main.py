import sys
import time

# main(sys.argv[1:])
if __name__ == "__main__":
    start_time = time.time()
    for i in range(10000):
        print(i)
    print("--- %s seconds ---" % (time.time() - start_time))
