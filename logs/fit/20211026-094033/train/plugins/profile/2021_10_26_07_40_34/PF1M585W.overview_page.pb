?($	?tMٿ??y??ֈ???!??u???!0?'???$	?? 2?@9???@p?H??Z@!{???$4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?!??u????L?J???A?/L?
F??Y]?Fx??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???H??w-!?l??A???{????YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??%䃞???W?2ı??Aw-!?l??Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???߾????W?2???A?d?`TR??Y8gDio??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0?'?????????A?):????Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ˡE????????3???A?\m?????Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vq?-???镲q??A?lV}????YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??n?????Fx$??Alxz?,C??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???H.??T㥛? ??A|a2U0*??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	<Nё\???jM??S??AQ?|a??Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??u??????????A?S㥛???Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<?R?!????3??7??A ?o_???Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Q?? ?~?:p??Affffff??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?2ı.??,e?X??A??Q????YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|гY????QI????Aw-!?l??Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O???uq???A??n????Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??S??+????A????o??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?#??????}?5^?I??Ao???T???Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?`TR'?????_vO??AZd;?O???Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D????9??A??ǘ???A??(???Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&OjM???F%u???A|a2U0??Y o?ŏ??*	     ?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatjM??St??!@hl_=J@)?=?U???15?rO#,I@:Preprocessing2F
Iterator::Model?<,Ԛ???!R?????6@)yX?5?;??1???Ĭ?,@:Preprocessing2U
Iterator::Model::ParallelMapV2?A`??"??!?XMH?g!@)?A`??"??1?XMH?g!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipǺ????!l _?OBS@)?I+???1'A??[?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??~j?t??! ?!?}?(@)?I+???1'A??[?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????Mb??!? ???@)????Mb??1? ???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??&S??!?????T1@)?c?ZB??1?A䁫h@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*9??v????!?`Z??@)9??v????1?`Z??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 ?h??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	^?I?i??hE3Y1???L?J???!w-!?l??	!       "	!       *	!       2$	?IZo????E7?Kz???/L?
F??!?):????:	!       B	!       J$	?dB$???????Q???<,Ԛ???!]?Fx??R	!       Z$	?dB$???????Q???<,Ԛ???!]?Fx??JCPU_ONLYY ?h??@b Y      Y@q???Gw??@"?
both?Your program is POTENTIALLY input-bound because 44.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.9823% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 