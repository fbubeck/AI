?&$	????0???_]?,?????/????!??????$	?p?÷?
@????o????? u?@!v1D?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Ӄ?R???????g?R??AByGsd??YR(__???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????????ԱJ??AǼ?8d??Yuʣa??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?;?%8??b?G??A???Z?a??YA,?9$???rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0[C??????}??????AX?x?a??Y_ѭ????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ꗈ?ο??rѬl??A?J??????Y???V???rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?>:u????n??ʆ5??A?-s?,&??Y+?`??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Z?$?9??_z?sѐ??A%Z?xZ~??Y???5w???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	???/?????x#????A	?3????Y??
G?J??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
O=???6??{j??U???A???W:??Y,???ؐ?rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?g)YN???-Y?&??A*8? "??Y\?	????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0E?u??2??ͮ{+??A????9??YD?l?????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????????Qf`??AN??1??Y???KUڒ?rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?~j?t??????|y??A??????Yt	?????rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0%??????"1?0??A5?uX??Y?????
??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?-?\??bۢ????A?ӹ????YM?]~??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?H??_?????9#J{??A?{?ʄ_??Y?Y??U???rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?1?????5
If???A3nj?????Y?VBwI???rtrain 97*	fffff??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?X???F??!?sI0e?B@)?????!??1?S???@@:Preprocessing2T
Iterator::Root::ParallelMapV2J?i?W??!???/?c0@)J?i?W??1???/?c0@:Preprocessing2E
Iterator::Rooth?RD?U??! k??a@@)?4S??1?Q;?_0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipC?5v????!?oJ??P@)&jj?Z??1?#???&@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice,-#??ʩ?!}}f?"@),-#??ʩ?1}}f?"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?9]???!??8p??,@)?.oך?1?n??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?????n??!?F>3ħ2@)~?.rO??11
??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor* C?*??!OY_.	@) C?*??1OY_.	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9'?jߣ
@IW???*X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	~"??b?????P?Ad???x#????!???ԱJ??	!       "	!       *	!       2$	vxd\???:?f?????*8? "??!ByGsd??:	!       B	!       J$	??zI???0???# W????5w???!?VBwI???R	!       Z$	??zI???0???# W????5w???!?VBwI???b	!       JCPU_ONLYY'?jߣ
@b qW???*X@Y      Y@q?n?$??S@"?	
both?Your program is POTENTIALLY input-bound because 46.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?78.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 