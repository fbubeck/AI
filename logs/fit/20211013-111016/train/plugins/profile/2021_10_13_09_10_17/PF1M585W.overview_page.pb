?($	???W#?????2????io???T??!c?ZB>???$	hӚ?@#28??i@???5?@!#?T>q+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$s??A?? c?ZB>??A???????Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&I.?!????????B???A????(??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W[??5?8EGr??A?T???N??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&1???vq?-??A??????Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
???1w-!??A??@?????YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S?????-?????Aj?t???Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?3??7?????ǘ????A?\m?????Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?3??7???j?t???AS?!?uq??Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W[???????z6???Aq=
ףp??Y_?Qڛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	X?2ı.?????_vO??A??0?*??Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
3ı.n???-???????A??_vO??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=,Ԛ????q??????A??D????Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z???xz?,C??AHP?s???Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ????sF????A6<?R???Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ŏ1w-!??lxz?,C??A?"??~j??Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P?s???????????Ah"lxz???Y??ݓ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
???ZӼ???A??9#J{??Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?G?z??J+???A)?Ǻ???Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}8????+e???A6?>W[???Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?ZB>????:M???A??ͪ????Y?:pΈҞ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&io???T??tF??_??AyX?5?;??Y???~?:??*	     ??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeato?ŏ1??!J3,??A@)??B?i???11?s@@:Preprocessing2F
Iterator::Modelr??????!s??2?@@)A??ǘ???1|뇛Y?4@:Preprocessing2U
Iterator::Model::ParallelMapV2F%u???!q????(@)F%u???1q????(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!sF???P@)Ԛ?????1???QA?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?W[?????!)lSwIy@)?W[?????1)lSwIy@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????߾??!?+8O,@)Tt$?????1U????$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??:M???!b?? ??3@)?~j?t???1
?b?Ϡ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Q?|a2??![?Q?L?@)Q?|a2??1[?Q?L?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?B7?l?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??4L?????F??w?? c?ZB>??!??ǘ????	!       "	!       *	!       2$	;??A????'?u:???yX?5?;??!6<?R???:	!       B	!       J$	2?A????>??nH?????Q???!??H?}??R	!       Z$	2?A????>??nH?????Q???!??H?}??JCPU_ONLYY?B7?l?@b Y      Y@q??g?.7G@"?
both?Your program is POTENTIALLY input-bound because 46.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?46.4311% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 