?($	FaA?!???????????	h"l??!???(\???$	??????@??ؓ? @NX??cO??!?ɯ?v:-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$$(~??k??v??????AW[??????Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&NbX9???F??_???A?Ǻ????Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M??St$?????????A$(~??k??Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0?'????I+???AKY?8????Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?l?????????(\???AQk?w????Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ё\?C?????y?):??A??H.???Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&5?8EGr???O??e??A??MbX??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S????*??D???A??^)??Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>?٬?\????ׁsF??A=?U?????Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	]?C?????㥛? ???AȘ?????Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
8gDio??????????AV????_??Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&?W??-C??6??Aŏ1w-??YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A??p_?Q??A???V?/??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ???A??ǘ???A??镲??Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(\????~?:p???ARI??&???Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<Nё\????|a2U??A?JY?8???Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&p_?Q??RI??&???A??<,Ԛ??Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ffffff??s??A??A?8??m4??Y??A?f??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX???T㥛? ??A? ?rh???Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q??????????<,??A?'????Y???B?i??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"l???Q?????Ah??s???Y333333??*	43333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?sF????!Z?`	e?B@)???1????1?d?HZEA@:Preprocessing2F
Iterator::Model?QI??&??!$+D?*?A@)??????1?&s?#7@:Preprocessing2U
Iterator::Model::ParallelMapV2????Q??!?_*,d?(@)????Q??1?_*,d?(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ݓ????!n?ݖ?"P@)?0?*??1? U`B?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice}гY????!?)po@)}гY????1?)po@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenateݵ?|г??!`"?>g)@)=?U?????1?H_@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Q?|??!???1@)?:pΈ??1????-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*/?$???!E??? @)/?$???1E??? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?k/5?
@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??d???$+?ω????Q?????!?~?:p???	!       "	!       *	!       2$	????_v??Ѩ? ְ?h??s???!RI??&???:	!       B	!       J$	;??3Eԙ??3?2?I??lxz?,C??!????o??R	!       Z$	;??3Eԙ??3?2?I??lxz?,C??!????o??JCPU_ONLYY?k/5?
@b Y      Y@q|86~@@"?
both?Your program is POTENTIALLY input-bound because 56.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?32.986% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 