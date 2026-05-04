

def python2_test_func(data_dict, output_file):
    print "Start processing..."   # Python2 print

    total = 0L   # long 类型

    # 遍历 dict（Python2 写法）
    for key, value in data_dict.iteritems():
        if value > 10:
            print key, value
            total += value
        elif value > 5:
            print "medium:", key
            total += value
        else:
            print "small:", key

    # xrange
    for i in xrange(3):
        print i

    # has_key
    if data_dict.has_key("error"):
        print "found error"

    # try-except-finally
    try:
        f = open(output_file, 'w')
        print >> f, "Result:", total   # Python2 重定向输出
    except IOError, e:
        print "IO Error:", e
        raise RuntimeError, "write failed"   # Python2 raise
    finally:
        try:
            f.close()
        except:
            pass

    return total
