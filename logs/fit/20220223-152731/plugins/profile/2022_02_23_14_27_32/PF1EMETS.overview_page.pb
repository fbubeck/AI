?&$	M/????????????4?Ry;??!	4??yT??$	uې??@?d|????1!??@!????;@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails07p??G??*?dq????A??)???Y㈵? ??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?6?^?7??g?????A?ʼUס??Yg???ْ?rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????Kq???Ӟ?sb??A??p?Qe??Y`??5!???rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0h??W??i?V????A?V?I???YƧ Ϡ??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?-?\???A?;???AIڍ>???YJ
,?)??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?[t???????fH??A???2???Y??KK??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?4?Ry;??2 Tq???A?!q????Y???&M???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?^}<?????'?$????A?]??a???Y?`?????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
͔?????)??Pj/??A?V횐??Yv???rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?j?=&R?????͋??A?8h???Y??̔?ߒ?rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?"??????q? ????A?^`V(???Y?9}=_??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Vc	kc????WuV??A??$"????Y(__?R??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????????lu9% ??AްmQf???YkD0.??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?O ????0?GQg???A?5!?1???Y??C?b??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?<?+J	??T:X??0??A?-??T??Y????W??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?!Y????0|DL??A?[??AA??YX???!??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	4??yT???6??nf??A?CQ?O???Y!??F???rtrain 97*	}?5^??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?KS8???!????[sC@)????????1n??-)bA@:Preprocessing2T
Iterator::Root::ParallelMapV2??ۻ??!????u/@)??ۻ??1????u/@:Preprocessing2E
Iterator::Root???????!???:??>@)??)???1ie?^?.@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice-^,?ӳ?!?-?&?	'@)-^,?ӳ?1?-?&?	'@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?$\?#???!?_qEDQ@)??iܛ߰?1.?H?Q?#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateAJ?i???!??Y?A0@)(`;?O??1Qݒ?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*k(??v??!&?=ŕ?@)k(??v??1&?=ŕ?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapx?ܙ	???![???\4@)?r?m?B??1???Fk@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??(c4%@It??\?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$		A
??ݜ???O??q? ????!A?;???	!       "	!       *	!       2$	?oz???????E????!q????!?CQ?O???:	!       B	!       J$	?'??9??6v???p?Ƨ Ϡ??!?`?????R	!       Z$	?'??9??6v???p?Ƨ Ϡ??!?`?????b	!       JCPU_ONLYY??(c4%@b qt??\?X@Y      Y@q[?,3?P@"?	
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
Refer to the TF2 Profiler FAQb?66.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 