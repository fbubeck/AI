?($	?ub??????|gx??z6?>W??!TR'?????$	xƵE?@YݮR0@[@??@!???5?I2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$z6?>W??Zd;?O???A??|?5^??Y???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ӽ?????JY?8???A`??"????Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&w??/???bX9????AF????x??YZd;?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~??	?c?Z??A?}8gD??Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF???d?`TR??A2U0*???Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????|a2U0*??A??n????Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W[??ŏ1w-!??A??y???Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<Nё\???8gDio???A?3??7??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????Έ?????A?&1???Y??#?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??h o???M??St$??A*:??H??YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ё\?C???6<?R???ADio?????Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Pk?w???V????_??A)??0???Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?rh??|??;M?O??A?S㥛???Yo?ŏ1??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2???1??%???A?W[?????Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ǘ???????(\???A?c]?F??Yf??a?֤?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????9#??Dio?????A:??H???Y_?Qڛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ݵ?|г??(??y??A?uq???Y8gDio??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????????x???A{?/L?
??Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??<,Ԛ??Y?8??m??AR'??????Y?N@aã?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&TR'???????y???AU0*????Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ??NbX9???A?Zd;???YDio??ɤ?*	?????%?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?(\?????!=?qd?@@)???V?/??1>Li???@:Preprocessing2F
Iterator::Model??a??4??!??B??B@)?=yX?5??1t?c???7@:Preprocessing2U
Iterator::Model::ParallelMapV2?J?4??!?C??+@)?J?4??1?C??+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?}8gD??!Wk?B?4O@)x$(~??1???u'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?;Nё\??!-K?4??@)?;Nё\??1-K?4??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?ZӼ???!^R "&?'@)>yX?5ͫ?1?Yȃ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap鷯???!4;?hs?0@)5?8EGr??1H?]??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??e?c]??!?At??@)??e?c]??1?At??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9s?'fd@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	 ?A?;)???GVG????Zd;?O???!??y???	!       "	!       *	!       2$	??W?c?????N??????|?5^??!U0*????:	!       B	!       J$	?K?R?Š?????&??ŏ1w-!??!???N@??R	!       Z$	?K?R?Š?????&??ŏ1w-!??!???N@??JCPU_ONLYYs?'fd@b Y      Y@qи???>@"?
both?Your program is POTENTIALLY input-bound because 54.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?30.7972% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 