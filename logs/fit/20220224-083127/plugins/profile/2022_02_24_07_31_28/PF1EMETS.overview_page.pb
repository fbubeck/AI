?&$	^!@????+?i?
??0?????!ٲ|]???$	?*ka!X@M??,????CP?j@!??7?b@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Z?wg????!??T2??A	Q????Y?KR?b??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0W?o????.?KR???A????(@??Y??i?????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?B??C?????]/M??As?蜟???Yj?@+0d??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????D???Or?Md??A9d?b???Y-?"?J ??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??e??t??j????4??A???E?n??YM֨?h??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0.s??/???&jin???A9??????Y???8Q??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0o?m???ADj??4??A7e??Y'3?Vzm??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??=x?R???6?^???Ad;?O????Y?? ???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?4c?t??? ?_>Y1??A4??????Y???(??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0I?\߇??|
????Al?f????Y?n/i?֑?rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails00??????p????A??*3????Y31]????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0~?Az?????,z???Af??a????Y????g???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0D4?????JEc??l??ARb??vK??Y9??!??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails02>?^????.????A"??T2 ??Y.Ȗ????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0C?=?????dT??A̶?ֈ`??Y%@7n??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?9x&4I?????rf??A??????Y??X ??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ٲ|]???5??.???Ao+?6??Y?;?2TŔ?rtrain 97*	?S㥛=?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?w?*??!e?6?B@)?@?"??14`?B?A@:Preprocessing2T
Iterator::Root::ParallelMapV2xԘsI??!q???q?1@)xԘsI??1q???q?1@:Preprocessing2E
Iterator::Root0,?-X??!m?<A@)??el?f??1j4??ɑ0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipPS???"??!????aP@)??d#٫?1w?M??#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?'????!?}Řy)"@)?'????1?}Řy)"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?=?Е??!ge)t&-@)????	??1??!??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapJ???nI??!??氜2@)?Nw?xΖ?1U??G?%@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*c
?8????!{</r@)c
?8????1{</r@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9&??@?1@Iߘ??rX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?z#?????tN?\???p????!?.?KR???	!       "	!       *	!       2$	y??MVT??????06??????(@??!o+?6??:	!       B	!       J$	*?υ???????hof?????g???!??i?????R	!       Z$	*?υ???????hof?????g???!??i?????b	!       JCPU_ONLYY&??@?1@b qߘ??rX@Y      Y@q?Ѣ???O@"?	
both?Your program is POTENTIALLY input-bound because 44.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?63.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 