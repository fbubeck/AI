?&$	?1?Z???G???j???dT8???!???G6W??$	y??߶@???#??-~??`@!?????@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???h?x??-?i??&??A?GߤiP??Y????vܐ?rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?D?$]3??????E??AOWw,?I??Y"S>U???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0dT8????j?3??A	O??'???Yaq8??9??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?,??\n????d?VA??A??o?N??Y????je??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0" 8????????iO???Av??^
??Y`#I????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??cx?????A??v???A?q??>s??Yȴ6?????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???K????h㈵???A]N	?I???YK?ó??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	L??pvk??.T???r??AkH?c?C??Y?o%;6??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
???????{,}????ApxADj???Y]?C?????rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0m?i?*????#????A??~?Ϛ??Y>Y1\ ??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???;????T?????Ax)u?8F??YB???ϝ??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0tϺFˁ??
??$>w??A?J???>??Y?_??s??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?3??????D?ÖM??A4?27߈??Y?ʼUס??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???G6W???1??8??A?W}??Y><K?P??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails03?Vzm????c?~???A{/?h???Y7?????rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0; ??^E?????9d??AK?8?????Y:\?=셒?rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?@+0du???3g}?1??As??A??Y?M?????rtrain 97*	=
ףp&?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat`???~???!???]TB@)?m??fc??1??ꬥ@@:Preprocessing2E
Iterator::Root??;?(A??!EP?pM?@@)A??h:;??1?a?V??1@:Preprocessing2T
Iterator::Root::ParallelMapV2?:?G??!P} 1J.@)?:?G??1P} 1J.@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?C?H????!A	??=?%@)?C?H????1A	??=?%@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?q?
??!?W?GY?P@)?QI??&??1??>???"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate???6????!?W??r1@)???$xC??1.L????@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??cx?g??!????-?4@)/?$????1??;??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?ZH????!FZ?
?
@)?ZH????1FZ?
?
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?6?W?@IL&?GX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	h?b?҉??R?|E?c?????iO???!?{,}????	!       "	!       *	!       2$	??`S?H????`Z?Ơ?v??^
??!?W}??:	!       B	!       J$	w!	??3??X?ӫsu?aq8??9??!`#I????R	!       Z$	w!	??3??X?ӫsu?aq8??9??!`#I????b	!       JCPU_ONLYY?6?W?@b qL&?GX@Y      Y@q&ɝ???U@"?	
both?Your program is POTENTIALLY input-bound because 44.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?86.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 