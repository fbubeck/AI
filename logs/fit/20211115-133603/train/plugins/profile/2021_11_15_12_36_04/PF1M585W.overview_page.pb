?($	?2???a?????>w????,C????!?9#J{???$	?<?h,@?3???	@?֬XMc@!??cwh?0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ͪ????????x???A&S????YTt$?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??4?8E???lV}????A;?O??n??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio???V????_??AjM??S??YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a??+e????ͪ????A?@??ǘ??Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{???m???{???A??	h"??Y??镲??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??q????5^?I??AGx$(??Y?:pΈҞ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C??6??;M?O??A$???~???Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S????䃞ͪ???A?f??j+??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&䃞ͪ???|a2U??AM?O????YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	8gDio??J{?/L???A[Ӽ???Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
m????????(??0??A?c?]K???Y??#?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??^)???/?'??A?MbX9??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&?W??P??n???A?	?c??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&:#J{?/?????1????ANbX9???YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~????ׁsF??A?3??7???Y_?Qڋ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF????v????AT㥛? ??Y_?Qڋ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P??n????H?}8??A      ??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??V?/????v??/??A?^)????Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k+??ݓ????ܵ?|??Aŏ1w-??Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???V?/??X?5?;N??A?n?????Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?,C????8??d?`??A&䃞ͪ??YK?=?U??*	????̀?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeato?ŏ1??!	????@@)W[??????1?Y?@?@:Preprocessing2F
Iterator::Model?H.?!???!??y?f?@@)???????1B?x?[4@:Preprocessing2U
Iterator::Model::ParallelMapV2??? ?r??!??(+?*@)??? ?r??1??(+?*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipd;?O????!?C?L?P@)?,C????1?&?Ss?(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?%䃞ͺ?!w??L-@)t$???~??1?*H?E?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?&1???!>?yST@)?&1???1>?yST@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_?L???!???H4@)?0?*???1p$?T@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?0?*???!p$?T@)?0?*???1p$?T@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9)?'%@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	n??:m??}g?k????8??d?`??!m???{???	!       "	!       *	!       2$	?_(?????,?kkk???&䃞ͪ??!Gx$(??:	!       B	!       J$	?J??_???&I?????_?Qڋ?!Tt$?????R	!       Z$	?J??_???&I?????_?Qڋ?!Tt$?????JCPU_ONLYY)?'%@b Y      Y@q?I?3C@"?
both?Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?38.1422% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 