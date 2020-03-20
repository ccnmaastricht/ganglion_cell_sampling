import RCA

inpath = 'C:/Users/Danny/PycharmProjects/RCA/in_folder'
outpath = 'C:/Users/Danny/PycharmProjects/RCA/out_folder'

RCA1 = RCA.RetinalCompression()

RCA1.series_dist(inpath, outpath)