?&$	??<-????a=I????o??R???!??Osr??$	?vu???@6$?{???t???U
@!oJe/@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Osr??ux??q??AX}w+??Y?%??s|??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?B?Y?????AA)Z???A???0`??YO?C?ͩ??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0lЗ??\???+J	????A?XQ?i??Y%?S;Ô?rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0v?r????ĵ??^(??A????x???Y??h o???rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0r?Z|
????D2????A?@+0du??Y@1?d???rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0~W?[???"??gx???A??!? ???Y???N????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?t<f?2???`?>??A??????Y?y0H???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?Q?????WC?K??AQ?|a2??Y??'?H0??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?ٕ?????0??????A	m9????YMۿ?Ҥ??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?o*Ral??ݔ?Z	???Ax^*6?u??YGY???.??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Z?A??v????m??A?cϞ???Y?{h+??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?i4?????3?ތ???A??????Y????_Z??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Fx{????	.V?`??AN?????Y?ɐ??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0t?f????? 6 B\9??Abe4?y???Y%\?#????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?o??R???Z?'??&??A3??O@??Y1`?U,~??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0g?ܶo??0?AC???AH???=??Y%t??Y??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0pUjv??5?\??u??A?3???Y>#?ƕ?rtrain 97*	ףp=
ׁ@2E
Iterator::Root?F????!??ݡ??F@)IIC????1'?%?=@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatM??u???!4?u?>@)u?BY????1BV?-?<@:Preprocessing2T
Iterator::Root::ParallelMapV2"?^F?ܶ?!?-(?jI/@)"?^F?ܶ?1?-(?jI/@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??\QJ??!?p"^jK@)q??#??1E?$ ?? @:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?*5{???!?[P??z @)?*5{???1?[P??z @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?*8? ??!?????(@)??=??W??1@8/	?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapE??@J???!V?-(?^/@)1}?!8.??1~?S)??
@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?}t??g??!??+f9b@)?}t??g??1??+f9b@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????h?@I³?s9?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	1?1rh????Y?H???Z?'??&??!ux??q??	!       "	!       *	!       2$	w??@z.??????3??O@??!?XQ?i??:	!       B	!       J$	???Z??ԡ?<3?u?%\?#????!%t??Y??R	!       Z$	???Z??ԡ?<3?u?%\?#????!%t??Y??b	!       JCPU_ONLYY????h?@b q³?s9?W@Y      Y@q?w??A?S@"?	
both?Your program is POTENTIALLY input-bound because 46.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?79.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 