?($	???d?Z??M ?$????c?]K???!Gx$(??$	~???E?@@?@??CU[	@!*?7d\?)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?L?J????s??˾?A<Nё\???Y?C??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/L?
F??ŏ1w-!??A??b?=??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio????N@a???A???????Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ǘ??????X????A????????Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#J{?/L???Fx$??Aj?q?????Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM??ݵ?|г??AZd;?O???Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?):?????[ A???A????o??Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O?????+e???A$(~??k??Y???JY???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L7?A`???????????Ar??????Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??s??????????A?j+?????YǺ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?7??d??????z6??A?[ A?c??Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??St???z?G???A??0?*??Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D????9????:M???A?W?2ı??Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Gx$(??w??/???AD?l?????Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`vOj???&1???A?;Nё\??YHP?sע?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&A??ǘ?????	h"l??AB?f??j??Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ڬ?\m?????6???A?=?U???Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ޓ??Z????ͪ??V??A?3??7???Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0L?
F%???e??a???A?s????YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?MbX9??vq?-??A??ݓ????Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?]K?????MbX??Ah??|?5??Yc?ZB>???*	?????K?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatO??e???!??|?A?@@)?G?z??1??3f?2?@:Preprocessing2F
Iterator::Model??ׁsF??!*:?;~A@)??0?*??1?}=??4@:Preprocessing2U
Iterator::Model::ParallelMapV2????Mb??!Qim^?E,@)????Mb??1Qim^?E,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipf??a????!k?b,?@P@)4??@????1?Y?~?$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenateio???T??!f?|??-@)?m4??@??1R??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceq???h??!{?_c?P@)q???h??1{?_c?P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapc?=yX??!>}??E5@)???߾??1-?@?&5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?z6?>??!?6/?"@)?z6?>??1?6/?"@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9s???E@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??%#~????<d?????MbX??!?[ A???	!       "	!       *	!       2$	l?Lڨ?????????h??|?5??!D?l?????:	!       B	!       J$	??h??7??'?<t?^K?=???!?C??????R	!       Z$	??h??7??'?<t?^K?=???!?C??????JCPU_ONLYYs???E@b Y      Y@q?"??hF@"?
both?Your program is POTENTIALLY input-bound because 49.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?44.8186% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 