LANGS = "af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur " \
        "vi yo zh".split()
pairs = []
for lg in LANGS:
    pairs.append("en-{}".format(lg))
    pairs.append("{}-en".format(lg))

print(",".join(pairs))
