?&$		.?:;???6$/F???~NA???!?*2: ???$	5????.@??O?V???j??

@!|,??o?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?)?:?????m?2??A?2?g??YZ??????rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?~NA???p"??????A? ?m?8??Y??J?????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??w?-;??:>Z?1???A?)?t??Y?ݒ????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????5??c'??>??A??rf?B??YR?GT???rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0,.??MT??h=|?(B??A+???+??Y???Oա?rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?*2: ???O??D????A?O?eo??Y8???????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?%????RI??&B??A????!9??Y?0?*???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??????erjg????A????B???Y??@?S??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
???1???>?h????APR`L??Y?ZӼ???rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??T?Gb??ǠB]??A?p????YfM,?ݢ?rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?f?"]??6!?1????A???????Y??j?	???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?T??? ???&k?C4??A??D2??Y?
???Ӧ?rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?t=?u????b?J!???A'?UH?I??Y??????rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0]?wb???0???"??A@?:s	??Y???????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?R??c???]3?f???A?^?D??Y??k]j???rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Fy?????????|@???A?y?W??Y?q??????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?1=a????&?`6??A[{C??Y(??h????rtrain 97*	?z?Gې@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??ݯ|??!?/?tB@)?z??&3??1qMz???@@:Preprocessing2E
Iterator::Root?,?s???!?z-??@)46<???1?D???w0@:Preprocessing2T
Iterator::Root::ParallelMapV2?%W??M??!?k?6?.@)?%W??M??1?k?6?.@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip\?nK????!E????Q@)w?$$?6??1.?ݕ?{'@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice,??ŷ?!?F??6!@),??ŷ?1?F??6!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenatePVW@??!yR?>T-@)???􃺰?1赋:@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap}Yک????!????s3@)?8K?r??1???4'@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?p!??F??!?JW??x
@)?p!??F??1?JW??x
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??UP?@I??Z?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	Xq??0????Y??0???"??!RI??&B??	!       "	!       *	!       2$	}??g9??{*?h??? ?m?8??!?O?eo??:	!       B	!       J$	Vm??뎡??P?$?t?(??h????!?
???Ӧ?R	!       Z$	Vm??뎡??P?$?t?(??h????!?
???Ӧ?b	!       JCPU_ONLYY??UP?@b q??Z?W@Y      Y@q??Y?B@"?	
both?Your program is POTENTIALLY input-bound because 47.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?37.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 