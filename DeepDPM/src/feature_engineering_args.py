import argparse

def add_model_specific_args(parser):
    # Community parameters
    parser.add_argument("--True_k",'--N',
                      type=int, default=8,
                      help='Number of isolates in the community (true number of clusters).'
                      )
    parser.add_argument('--S',
                      type=int, default=99,
                      help='Seed set for the creation of the community'
                      )
    parser.add_argument('--is_balanced',
                      default="balanced",
                      choices=["balanced","unbalanced"],
                      help='Whether the community is balanced or not (number of reads per isolate is the same).'
                      )

    # FCGR parameters
    parser.add_argument('--kmer',
                      type=int, default=5,
                      help='k-mer size (4 is tetranucleotide, 5 is pentanucleotide, etc.)'
                      )
    parser.add_argument('--IPD_stride',
                      type=str, default="max",
                      help='the method to use for the IPD stride (normal, mean, max, emph_A)'
                      )
    parser.add_argument('--IPD_r',
                      type=float, default=1,
                      help='when striding along the 3rd axis (with the IPD values), the strep size is multiplied by this value. Probably either 0.5 or 1'
                      )
    parser.add_argument('--n_channels','--channels',
                      type=int, default=3, choices=[1,3]
                      #help="Number of channels in the FCGR (1 is mono-channel, 3 is RGB). Mono-channel doesnt cosider IPDs."
                      )
    parser.add_argument('--IPD_stratification',
                      type=str, default="MeanStd",
                      help='the method to use for the splitting the Z axis into groups (MeanStd or FullRange)'
                      )

    # CGR parameters
    parser.add_argument('--read_length',
                      type=int, default=10_000,
                      help='The threshold by which the reads were filteted (i.e. reads shorter than this value were discarded).'
                      )
    parser.add_argument('--effective_coverage',
                      type=float, default=10.0,
                      help='The threshold by which the reads were filteted (i.e. reads with effective coverage lower than this value were discarded).'
                      )

    return parser