?($	??"???????Ɛ???j?t???!i o????$	b??S??@??G??@?d?7?? @!???3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?St$?????&?W??AY?? ???Y	??g????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?c???d?]K???A??A?f??Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2ı??}гY????AR'??????Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&i o??????_?L??A??&???Y?N@aã?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???{????f?c]?F??A6?;Nё??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J{?/L????x?&1??A??~j?t??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#??~j???䃞ͪ???A??S㥛??Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&e?X???O??e?c??ApΈ?????YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'???????      ??A??D????Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	^K?=?????????A?????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
46<????<,Ԛ???A??_?L??Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&w??/???x$(~??A?ݓ??Z??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ???I.?!????A?QI??&??Y2??%䃎?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"lxz?,??M?O???AZd;?O???YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&HP?s???RI??&???A?0?*??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ZӼ???f??a????A333333??Y_?Qڋ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&؁sF????x$(~??AGr?????YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??9#J{?????T????AJ{?/L???YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??x?&1???'????Av??????YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&i o?????h o???A]?C?????Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t?????q????Ap_?Q??YM??St$??*	?????%?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ڊ?e???!???G???@)?&S???1f?n]=@:Preprocessing2F
Iterator::Model??_?L??!?7b??C@)???1????1????%?7@:Preprocessing2U
Iterator::Model::ParallelMapV21?*?Թ?!y ??y,@)1?*?Թ?1y ??y,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???߾??!rȝ?N@)0L?
F%??1???nO'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???<,Ԫ?!??vE?@)???<,Ԫ?1??vE?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?E???Ը?!ޱ|??_+@)??ͪ?զ?1??O@,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~j?t???!v?b?E2@)?p=
ף??1?u?W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?&S???!T???n?@)?&S???1T???n?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?$$nr@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	n????????N?????q????!d?]K???	!       "	!       *	!       2$	H??????Uu??ٸ??p_?Q??!??&???:	!       B	!       J$	U?OШ???2}?a?k??S?!?uq??!	??g????R	!       Z$	U?OШ???2}?a?k??S?!?uq??!	??g????JCPU_ONLYY?$$nr@b Y      Y@qX????H@"?
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?49.3885% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 