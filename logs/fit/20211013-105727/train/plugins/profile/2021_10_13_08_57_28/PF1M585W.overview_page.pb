?)$	e}???????;G?????t?V??!??3?4 @$	?2?s@?2??}?@??????@!??q?*6:@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ׁsF???C??????AF????x??YH?}8g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????K7???E??????A??St$???Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z??????A?*??	??Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??k	????:??H???Ac?=yX??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&5^?I??=?U????A??_?L??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Q????&S??A?&S???YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A`??"??q???h??A$???????YǺ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~?:p???"??u????A??7??d??Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-???,e?X??A)??0???Y?]K?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?????M??????????A?? ???Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??A?f????????A#??~j???Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&OjM???\ A?c???A=
ףp=??Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???JY???o??ʡ??A"??u????Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q=
ףp????/?$??A?3??7???Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&EGr?????s????A??j+????Y㥛? ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?;Nё??鷯???A??ʡE??Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??3?4 @????o??A?!??u???Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ˡE?????? ?	???A?z?G???Y(~??k	??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Tt$??????9#J{???A?q?????YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&v??????jM??St??A;M?O??YI.?!????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?t?V???b?=y??A?}8gD??Y|??Pk???*	fffffԔ@2F
Iterator::ModelԚ?????!?????J@)??0?*??1+E??JE@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeats??A???!??	??:@)??u????1?O???8@:Preprocessing2U
Iterator::Model::ParallelMapV2?3??7???!?F?;&@)?3??7???1?F?;&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??6???!N?:6(&G@)g??j+???1~???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?s????!???O?@)?s????1???O?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?T???N??!}??*#@)-!?lV??1	?i1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??A?f??!??^??)@)8??d?`??1p`V?Y?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*_?Qڛ?!YQ<??R @)_?Qڛ?1YQ<??R @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t51.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??T?"@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	@R[>'????}???>???b?=y??!????o??	!       "	!       *	!       2$	???Nt???K}QF,o???}8gD??!F????x??:	!       B	!       J$	???????Gz?)y??ݵ?|г??!H?}8g??R	!       Z$	???????Gz?)y??ݵ?|г??!H?}8g??JCPU_ONLYY??T?"@b Y      Y@qHu?/~@@"?	
both?Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t51.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?32.0664% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 