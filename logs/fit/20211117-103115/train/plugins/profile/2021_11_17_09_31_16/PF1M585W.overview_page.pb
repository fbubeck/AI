?($	??? ?l??T???IT????	h"l??!O??e???$	 ?w?@s?q?@J	@|??n??@!P)??jh1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??1??%????&S??Ai o????Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?=?U????Q???A??|?5^??Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??g??s???^)???A??_vO??Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C??6??-??????A?MbX9??YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C?i?q???_?L?J??A|a2U0*??Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e???e?`TR'??A??Q????Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??_?L?????<,???A??镲??Y	??g????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??z6???j?t???A????Q??YǺ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$???~?????_vO??Affffff??YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	O@a????ۊ?e????A???߾??Y?]K?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??y?):???????B??A???x?&??Y?|a2U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???9#J??e?X???AJ{?/L???Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?):?????g??s???A?6?[ ??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ǘ?????Έ?????A?&S???YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&bX9????h??|?5??AyX?5?;??Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J{?/L???M?O????A)??0???Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2U0*???`vOj??A5?8EGr??YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<???|??Pk???AM?J???YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ffffff???W?2ı??A??ׁsF??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&gDio????mV}??b??A???_vO??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"l???HP???A?|a2U??Y??H?}??*	????̠?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?U??????!B"???A@)|a2U0??18?Sͥ@@:Preprocessing2F
Iterator::Model}гY????!-?2KC?@@)????_v??1?????4@:Preprocessing2U
Iterator::Model::ParallelMapV2??s????!Wkȹ(@)??s????1Wkȹ(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip!?rh????!??fZ޹P@)????9#??1~ ??'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?6?[ ??!ٕA/?D@)?6?[ ??1ٕA/?D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?E??????!l?4?m?-@)aTR'????1??';:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapOjM???! ????G4@)?b?=y??1?]5%c?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??ݓ????!??V??o@)??ݓ????1??V??o@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9B?6??x@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	P?2????*?J??!???HP???!???<,???	!       "	!       *	!       2$	'W?D???B??}????|a2U??!??Q????:	!       B	!       J$	???_???{]??Ą????QI??!???߾??R	!       Z$	???_???{]??Ą????QI??!???߾??JCPU_ONLYYB?6??x@b Y      Y@q?.??V??@"?
both?Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.7943% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 