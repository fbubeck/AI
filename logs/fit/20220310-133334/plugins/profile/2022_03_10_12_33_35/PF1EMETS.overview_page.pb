?&$	???q?5??EӮTR??1	???!??$w????$	??H??@3?"%??eT??L@!!?3?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Ȳ`??"??;V)=?K??A?J???Y????g???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??$w?????].?;1??Ay?ՏM??YX?eS???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Qew???ˡE?????A?r.?U??Y??Os?"??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ș&l????$
-?????A???%:???Y?+?S???rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?f?????`???p??A?????Q??Y?Ko.??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0t?//?>???;? ??A????Kq??Y?Ia??L??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????K??(b???Al??3?I??Y???6?h??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?$??r???tYLl>??A:ZՒ?r??Y?QcB?%??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
1	?????x>???Aڎ?????Y???q???rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails06???X??????S???A,??ص???Y?W?ۼ??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0W^??????d> Й???A6>???4??Y^?????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0}"O???????߃?.??A?9? U??Y??ao??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??ŉ????=??M???A?\n0?a??Y?Hڍ>???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?a??A???OT6????A?$????Yӿ$?)???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???8??J?????ASu?l????Y?dȱ???rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?GT?n??????j?=??AR+L?k??YG?P?[??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0㊋?????q4GV~??A???N@??Y?GnM?-??rtrain 97*	V-???@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatfM,????!? /B@)???????1A#8?vg@@:Preprocessing2E
Iterator::Root???v?>??!t?E/?A@)?PMI????1?z???"2@:Preprocessing2T
Iterator::Root::ParallelMapV2{L?4???!Ԭw41@){L?4???1Ԭw41@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip+??-??!G]h@*P@)?'?_??1?y??^?#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??`<??!???w??#@)??`<??1???w??#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?~??@???!([E?Y.@)1?߄B??17?<[k@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*.??Hْ?!?7??y@).??Hْ?1?7??y@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap_Cp\?M??!???1q2@)??????1D?V/&L@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no93#l˻@I~瞤!"X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	؊?-i????^??????x>???!?].?;1??	!       "	!       *	!       2$	??????Y??nr????$????!,??ص???:	!       B	!       J$	Mq???Y???t?g?X?eS???!???6?h??R	!       Z$	Mq???Y???t?g?X?eS???!???6?h??b	!       JCPU_ONLYY3#l˻@b q~瞤!"X@Y      Y@q?du{?EU@"?	
both?Your program is POTENTIALLY input-bound because 44.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?85.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 