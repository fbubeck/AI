?($	Ϻ+&g???c05???}?5^?I??!?y?):?@$	?;?k?@???>ZJ@?'?@!?@??,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?X????i o????A%??C???Yio???T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v???bX9????A??ܵ??Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????w??/???A5^?I??Yd?]K???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A?f????ݵ?|г??A??K7?A??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&,e?X??xz?,C??A?=?U???Y?lV}????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?@z?):????A?(\?????YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?y?):?@?L?J???AOjM???YY?8??m??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&     ?@%??C???A?&?W??Y?i?q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&fffff? @|??Pk???A??|?5^??Y?8??m4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	
h"lxz???(\?????A????K7??Y??H.?!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??N@a???X?? ??A鷯????Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&؁sF?v @=,Ԛ???A?O??n??Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~j?t???0L?
F%??Aj?q?????Y?rh??|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??V?/?????_?L??A4??@????Y?A?fշ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-??o@?ݓ??Z??A??(????Yŏ1w-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?E???T@?G?z???A?Y??ڊ??Y?k	??g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|?5^??J{?/L???Aj?t???Y?U???د?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&aTR'????$(~????A?Zd;??Y???&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??3???????H.??A?m4??@??Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????c]?F??A$(~??k??Y??(???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}?5^?I??R???Q??A?镲q??Yz?):?˯?*	effffJ?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat	?c?Z??!e?????B@)Gx$(??1?o_??tA@:Preprocessing2F
Iterator::Model????9#??!?w??qA@)$???~???1t
?l?4@:Preprocessing2U
Iterator::Model::ParallelMapV2?D???J??!?????k,@)?D???J??1?????k,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?[ A?c??!%D??)GP@)A??ǘ???1?pQǂ$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?H?}8??!?H???@)?H?}8??1?H???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateTt$?????!ׅ|?;?(@)O@a?ӻ?1??x#Y?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Q?|??!d??$1@)ꕲq???1?Y?x@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*6<?R???!ߨg֢y@)6<?R???1ߨg֢y@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?גI~?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	y?)?*????,??????R???Q??!?L?J???	!       "	!       *	!       2$	?Pk?w???Td??????镲q??!OjM???:	!       B	!       J$	?_ļ?K??-??Ce??d?]K???!io???T??R	!       Z$	?_ļ?K??-??Ce??d?]K???!io???T??JCPU_ONLYY?גI~?@b Y      Y@qD&?6??U@"?
both?Your program is POTENTIALLY input-bound because 55.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?87.8742% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 