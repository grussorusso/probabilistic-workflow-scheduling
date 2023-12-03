from scheduler.job import *
from numpy import random

def create_simple_job ():
    op1 = Operator("OpA")
    op2 = Operator("OpB")
    op3 = Operator("OpC")
    job = Job()
    job.add_node(op1)
    job.add_node(op2)
    job.add_node(op3)
    job.add_edge(op1,op2)
    job.add_edge(op1,op3)
    return job, None

def create_example_snap_job (max_operators=99):
    job = Job()
    r = Operator("Read", job)
    job.add_node(r)
    r2 = Operator("Read2", job)
    job.add_node(r2)
    s = [Operator(f"Split{i}", job) for i in range(6)]
    o = [Operator(f"Orbit{i}", job) for i in range(6)]
    c = [Operator(f"Calibration{i}", job) for i in range(6)]
    for i in range(0,3):
        job.add_node(s[i])
        job.add_node(c[i])
        job.add_node(o[i])
        job.add_edge(r,s[i])
        job.add_edge(s[i],o[i])
        job.add_edge(o[i],c[i])
    if len(job.nodes) >= max_operators:
        return job

    for i in range(3,6):
        job.add_node(s[i])
        job.add_node(c[i])
        job.add_node(o[i])
        job.add_edge(r2,s[i])
        job.add_edge(s[i],o[i])
        job.add_edge(o[i],c[i])
    if len(job.nodes) >= max_operators:
        return job
    #--
    gc = [Operator(f"geocoding{i}", job) for i in range(3)]
    inte = [Operator(f"interferogram{i}", job) for i in range(3)]
    tops = [Operator(f"deburst{i}", job) for i in range(3)]
    merge = Operator("merger", job)
    job.add_node(merge)
    for i in range(0,3):
        job.add_node(gc[i])
        job.add_node(inte[i])
        job.add_node(tops[i])
        job.add_edge(c[i],gc[i])
        job.add_edge(c[3+i],gc[i])
        job.add_edge(gc[i],inte[i])
        job.add_edge(inte[i],tops[i])
        job.add_edge(tops[i],merge)
    if len(job.nodes) >= max_operators:
        return job
    #--
    multilook = Operator("multilook", job)
    job.add_node(multilook)
    corr = Operator("terrainCorrection", job)
    job.add_node(corr)
    write = Operator("write", job)
    job.add_node(write)

    job.add_edge(merge,multilook)
    job.add_edge(multilook,corr)
    job.add_edge(corr,write)

    return job, None

def create_epigenomics (N_SPLITS=4):
    job = Job()
    fq = Operator("fastQsplit")
    job.add_node(fq)

    filters=[Operator(f"filter_{i}") for i in range(N_SPLITS)]
    sol=[Operator(f"sol_{i}") for i in range(N_SPLITS)]
    bfq=[Operator(f"bfq_{i}") for i in range(N_SPLITS)]
    maps=[Operator(f"maps_{i}") for i in range(N_SPLITS)]
    

    merge = Operator("mapMerge")
    job.add_node(merge)
    index = Operator("maqIndex")
    job.add_node(index)
    pileup = Operator("pileup")
    job.add_node(pileup)

    for i in range(N_SPLITS):
        job.add_node(filters[i])
        job.add_node(sol[i])
        job.add_node(bfq[i])
        job.add_node(maps[i])

        job.add_edge(fq, filters[i])
        job.add_edge(filters[i], sol[i])
        job.add_edge(sol[i], bfq[i])
        job.add_edge(bfq[i], maps[i])
        job.add_edge(maps[i], merge)

    job.add_edge(merge, index)
    job.add_edge(index, pileup)

    mean_durations = {}
    mean_durations[fq] = 34.32
    for i in range(N_SPLITS):
        mean_durations[filters[i]] = 2.47
        mean_durations[sol[i]] = 0.48
        mean_durations[bfq[i]] = 1.40
        mean_durations[maps[i]] = 201.89
    mean_durations[merge] = 11.01
    mean_durations[index] = 43.57
    mean_durations[pileup] = 55.95

    output_mb = {}
    output_mb[fq] = 242.29
    output_mb[pileup] = 83.95
    output_mb[merge] = 26.71
    for i in range(N_SPLITS):
        output_mb[filters[i]] = 13.25
        output_mb[sol[i]] = 10.09
        output_mb[bfq[i]] = 2.22
        output_mb[maps[i]] = 0.90
    output_mb[index]=107.53

    
    return job, mean_durations, output_mb

def create_montage ():
    N0 = 4
    N1 = 6
    NTBL = 2
    job = Job()
    projects = [Operator(f"project_{i}") for i in range(N0)]
    for op in projects:
        job.add_node(op)
    diffs = [Operator(f"diff_{i}") for i in range(N1)]
    for op in diffs:
        job.add_node(op)

    concat = Operator("concat")
    job.add_node(concat)

    bgmodel = Operator("bgmodel")
    job.add_node(bgmodel)

    backgrounds = [Operator(f"bg_{i}") for i in range(N0)]
    for op in backgrounds:
        job.add_node(op)

    mtbl = [Operator(f"mtbl_{i}") for i in range(NTBL)]
    for op in mtbl:
        job.add_node(op)
    add = [Operator(f"add_{i}") for i in range(NTBL)]
    for op in add:
        job.add_node(op)
    shrink = [Operator(f"shrink_{i}") for i in range(NTBL)]
    for op in shrink:
        job.add_node(op)
    

    globtbl = Operator("globtbl")
    job.add_node(globtbl)
    globadd = Operator("globadd")
    job.add_node(globadd)
    jpeg = Operator("jpeg")
    job.add_node(jpeg)

    for j in [0,1]:
        job.add_edge(projects[0], diffs[j])
    for j in [0,1,2,4]:
        job.add_edge(projects[1], diffs[j])
    for j in [2,3]:
        job.add_edge(projects[2], diffs[j])
    for j in [3,4,5]:
        job.add_edge(projects[3], diffs[j])
    
    for i in range(N0):
        job.add_edge(projects[i], backgrounds[i])

    for i in range(N1):
        job.add_edge(diffs[i], concat)
    job.add_edge(concat, bgmodel)
    for i in range(N0):
        job.add_edge(bgmodel, backgrounds[i])
        job.add_edge(backgrounds[i], mtbl[i%NTBL])
    for i in range(NTBL):
        job.add_edge(mtbl[i], add[i])
        job.add_edge(add[i], shrink[i])
        job.add_edge( shrink[i], globtbl)
    job.add_edge(globtbl, globadd)
    job.add_edge(globadd, jpeg)

    mean_durations = {}
    for i in range(N0):
        mean_durations[projects[i]] = 1.73
        mean_durations[backgrounds[i]] = 1.72
    for i in range(N1):
        mean_durations[diffs[i]] = 0.66
    mean_durations[concat] = 143.26
    mean_durations[bgmodel] = 384.49
    for i in range(NTBL):
        mean_durations[mtbl[i]] = 2.78
        mean_durations[add[i]] = 282.37
        mean_durations[shrink[i]] = 66.10
    mean_durations[globadd] = mean_durations[add[0]]
    mean_durations[globtbl] = mean_durations[mtbl[0]]
    mean_durations[jpeg] = 0.64

    output_mb = {}
    for i in range(N0):
        output_mb[projects[i]] = 8.09
        output_mb[backgrounds[i]] = 8.09
    for i in range(N1):
        output_mb[diffs[i]] = 0.64
    output_mb[concat] = 1.22
    output_mb[bgmodel] = 0.10
    for i in range(NTBL):
        output_mb[mtbl[i]] = 0.12
        output_mb[add[i]] = 775.45
        output_mb[shrink[i]] = 0.49
    output_mb[globadd] = output_mb[add[0]]
    output_mb[globtbl] = output_mb[mtbl[0]]
    output_mb[jpeg] = 0.39

    return job, mean_durations, output_mb

# See Fig.6, Calzarossa
def create_cybershake ():
    branches=[8,5]

    job = Job()
    extract = [Operator(f"extract_{i}") for i in range(2)]
    for op in extract:
        job.add_node(op)
    synthesis = [Operator(f"synth_{i}") for i in range(sum(branches))]
    for op in synthesis:
        job.add_node(op)
    peakval = [Operator(f"peak_{i}") for i in range(sum(branches))]
    for op in peakval:
        job.add_node(op)

    zip1 = Operator("zip1")
    job.add_node(zip1)

    zip2 = Operator("zip2")
    job.add_node(zip2)

    synt_i = 0
    for j in [0,1]:
        for i in range(branches[j]):
            job.add_edge(extract[j], synthesis[synt_i])
            job.add_edge(synthesis[synt_i], peakval[synt_i])
            job.add_edge(synthesis[synt_i], zip1)
            job.add_edge(peakval[synt_i], zip2)
            synt_i += 1


    mean_durations = {}
    for op in extract:
        mean_durations[op] = 110.58
    for op in synthesis:
        mean_durations[op] = 79.47
    mean_durations[zip1] = 265.73
    mean_durations[zip2] = 195.80
    for op in peakval:
        mean_durations[op] = 0.55

    output_mb = {}
    for op in extract:
        output_mb[op] = 155.86
    for op in synthesis:
        output_mb[op] = 0.02
    output_mb[zip1] = 101.05
    output_mb[zip2] = 2.26
    for op in peakval:
        output_mb[op] = 0.0001


    return job, mean_durations, output_mb

def create_sipht (n=18):

    job = Job()
    patser = [Operator(f"patser_{i}") for i in range(n)]
    for op in patser:
        job.add_node(op)

    transterm = Operator("transterm")
    job.add_node(transterm)
    findterm = Operator("findterm")
    job.add_node(findterm)
    rnamotif = Operator("rnamotif")
    job.add_node(rnamotif)
    blast = Operator("blast")
    job.add_node(blast)
    srna = Operator("srna")
    job.add_node(srna)
    ffnparse = Operator("ffnparse")
    job.add_node(ffnparse)
    blastsyn = Operator("blastsyn")
    job.add_node(blastsyn)
    blastcand = Operator("blastcand")
    job.add_node(blastcand)
    blastq = Operator("blastq")
    job.add_node(blastq)
    blastpara = Operator("blastpara")
    job.add_node(blastpara)
    annotate = Operator("annotate")
    job.add_node(annotate)
    concate = Operator("concate")
    job.add_node(concate)

    job.add_edge(transterm, srna)
    job.add_edge(findterm, srna)
    job.add_edge(rnamotif, srna)
    job.add_edge(blast, srna)

    job.add_edge(srna, ffnparse)
    job.add_edge(srna, blastsyn)
    job.add_edge(srna, blastq)
    job.add_edge(srna, blastpara)
    job.add_edge(srna, blastcand)
    job.add_edge(ffnparse, blastsyn)
    job.add_edge(blastsyn, annotate)
    job.add_edge(blastq, annotate)
    job.add_edge(blastpara, annotate)
    job.add_edge(blastcand, annotate)

    job.add_edge(concate, annotate)

    for i in range(n):
        job.add_edge(patser[i], concate)


    mean_durations = {}
    for op in patser:
        mean_durations[op] = 0.96
    mean_durations[concate] = 0.03
    mean_durations[transterm] = 32.41
    mean_durations[findterm] = 594.94
    mean_durations[rnamotif] = 25.69
    mean_durations[blast] = 3311.12
    mean_durations[srna] = 12.44
    mean_durations[ffnparse] = 0.73
    mean_durations[blastsyn] = 3.37
    mean_durations[blastcand] = 0.6
    mean_durations[blastq] = 440.88
    mean_durations[blastpara] = 0.68
    mean_durations[annotate] = 0.14
    
    output_mb = {}
    for op in patser:
        output_mb[op] = 0.00001
    output_mb[concate] = 0.14
    output_mb[transterm] = 0.00001
    output_mb[findterm] = 379.01
    output_mb[rnamotif] = 0.04
    output_mb[blast] = 565.06
    output_mb[srna] = 1.32
    output_mb[ffnparse] = 2.51
    output_mb[blastsyn] = 0.42
    output_mb[blastcand] = 0.06
    output_mb[blastq] = 567.01
    output_mb[blastpara] = 0.03
    output_mb[annotate] = 0.03


    return job, mean_durations, output_mb

# DOI: 10.1109/WORKS.2008.4723958, Fig. 5
def create_ligo ():
    n=9
    job = Job()
    tmplt = [Operator(f"tmplt_{i}") for i in range(n)]
    for op in tmplt:
        job.add_node(op)
    inspiral1 = [Operator(f"inspiral1_{i}") for i in range(n)]
    for op in inspiral1:
        job.add_node(op)
    trig = [Operator(f"trig_{i}") for i in range(n)]
    for op in trig:
        job.add_node(op)
    inspiral2 = [Operator(f"inspiral2_{i}") for i in range(n)]
    for op in inspiral2:
        job.add_node(op)

    thinca = [Operator(f"thinca_{i}") for i in range(4)]
    for op in thinca:
        job.add_node(op)


    for i in range(n):
        job.add_edge(tmplt[i], inspiral1[i])
        job.add_edge(trig[i], inspiral2[i])
    for i in range(5):
        job.add_edge(inspiral1[i], thinca[0])
        job.add_edge(inspiral2[i], thinca[2])
    for i in range(5,n):
        job.add_edge(inspiral1[i], thinca[1])
        job.add_edge(inspiral2[i], thinca[3])


    mean_durations = {}
    for op in tmplt:
        mean_durations[op] = 18.14
    for op in thinca:
        mean_durations[op] = 5.37
    for op in trig:
        mean_durations[op] = 5.11
    for op in inspiral1:
        mean_durations[op] = 460.21
    for op in inspiral2:
        mean_durations[op] = mean_durations[inspiral1[0]]

    output_mb = {}
    for op in tmplt:
        output_mb[op] = 0.94
    for op in thinca:
        output_mb[op] = 0.03
    for op in trig:
        output_mb[op] = 0.01
    for op in inspiral1:
        output_mb[op] = 0.30
    for op in inspiral2:
        output_mb[op] = output_mb[inspiral1[0]]
    
    return job, mean_durations, output_mb


def create_dummy_pipeline (tasks=5):
    rng = random.default_rng(tasks)
    job = Job()

    ops = [Operator(f"op_{i}") for i in range(tasks)]
    for op in ops:
        job.add_node(op)
    for i in range(tasks-1):
        job.add_edge(ops[i], ops[i+1])

    mean_durations = {}
    output_mb = {}
    for op in ops:
        mean_durations[op] = rng.uniform(1,100)
        output_mb[op] = rng.uniform(0.1,10)

    return job, mean_durations, output_mb

def create_dummy_fire_detection ():
    rng = random.default_rng(1)
    job = Job()


    N=6
    opsA = [Operator(f"opA_{i}") for i in range(N)]
    for op in opsA:
        job.add_node(op)
    for i in range(N-1):
        job.add_edge(opsA[i], opsA[i+1])
    opsB = [Operator(f"opB_{i}") for i in range(N)]
    for op in opsB:
        job.add_node(op)
    for i in range(N-1):
        job.add_edge(opsB[i], opsB[i+1])
    opsC = [Operator(f"opC_{i}") for i in range(N)]
    for op in opsC:
        job.add_node(op)
    for i in range(N-1):
        job.add_edge(opsC[i], opsC[i+1])

    N2=3
    opsD = [Operator(f"opD_{i}") for i in range(N2)]
    for op in opsD:
        job.add_node(op)
    for i in range(N2-1):
        job.add_edge(opsD[i], opsD[i+1])

    opExtra = Operator("extra")
    job.add_node(opExtra)
    all_ops = opsA+opsB+opsC+opsD+[opExtra]

    job.add_edge(opsA[-1], opsC[3])
    job.add_edge(opsB[-1], opsC[3])
    job.add_edge(opsD[-1], opsC[3])
    job.add_edge(opExtra, opsC[3])

    mean_durations = {}
    output_mb = {}
    for op in all_ops:
        mean_durations[op] = rng.uniform(1,100)
        output_mb[op] = rng.uniform(0.1,10)

    return job, mean_durations, output_mb
