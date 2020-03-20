import RCA

FOV = 25
in_path = 'C:/<Path to input folder>'
out_path = 'C:/<Path to output folder>'

RCA = RCA.RetinalCompression()
RCA.series(in_path, out_path)
